"""Train all-layer QLoRA on ten-image sets. Only answer tokens have a loss."""

import argparse
import json
import math
from pathlib import Path
import random
from build_training_sets import load
from common import *


def targets(model):
    import torch

    names = {id(module): name for name, module in model.named_modules()}
    result = []
    for block in layers(model):
        found = [
            names[id(block)] + "." + name
            for name, module in block.named_modules()
            if name and isinstance(module, torch.nn.Linear)
        ]
        if not found:
            raise ValueError("A decoder layer has no supported linear modules")
        result.extend(found)
    return result


def trainable(model, selected):
    names = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not names or any(
        "lora_" not in name or not any(t + "." in name for t in selected)
        for name in names
    ):
        raise ValueError(
            "Trainable parameters escape the selected decoder LoRA modules"
        )
    return dict(parameters=names, total=sum(names.values()), target_modules=selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_args(p)
    p.add_argument("--baseline-run", type=Path)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, choices=(2, 3), default=2)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--validation-frequency", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-tokens", type=int, default=16000)
    a = p.parse_args()
    m = load(a.manifest, a.data_dir)
    model_settings = settings(a)
    if a.adapter:
        p.error("Training starts from the pretrained base, not --adapter")
    if (
        min(
            a.rank,
            a.alpha,
            a.gradient_accumulation,
            a.validation_frequency,
            a.checkpoint_interval,
            a.max_tokens,
        )
        < 1
    ):
        p.error("Counts must be positive")
    if not 0 <= a.dropout < 1 or a.learning_rate <= 0:
        p.error("Invalid dropout or learning rate")
    if a.validate_only:
        print(
            "Validated 90 training and 30 validation sets. Test sets will not train or select checkpoints."
        )
        return
    if not a.smoke_test:
        if not a.baseline_run:
            p.error(
                "Run the pretrained validation evaluation first and supply --baseline-run"
            )
        base = json.loads((a.baseline_run / "config.json").read_text())
        result = json.loads((a.baseline_run / "metrics.json").read_text())
        if (
            base["settings"] != model_settings
            or base["manifest_hash"] != digest(m)
            or base["partition"] != "validation"
            or base["adapter"]
            or base["smoke_test"]
            or result["trials"] != 90
        ):
            raise ValueError(
                "Need complete pretrained validation evaluation with identical inputs/settings"
            )
    import torch
    from peft import (
        get_peft_model,
        LoraConfig,
        prepare_model_for_kbit_training,
        PeftModel,
    )

    torch.manual_seed(2026)
    random.seed(2026)
    adapter = load_model(a, training=True)
    chosen = targets(adapter.model)
    for parameter in adapter.model.parameters():
        parameter.requires_grad = False
    if a.precision == "4bit":
        adapter.model = prepare_model_for_kbit_training(
            adapter.model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        adapter.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if a.resume:
        info = json.loads((a.resume / "experiment.json").read_text())
        if info["settings"] != model_settings or info["manifest_hash"] != digest(m):
            raise ValueError("Checkpoint inputs/model differ")
        adapter.model = PeftModel.from_pretrained(
            adapter.model, a.resume, is_trainable=True
        )
    else:
        adapter.model = get_peft_model(
            adapter.model,
            LoraConfig(
                r=a.rank,
                lora_alpha=a.alpha,
                lora_dropout=a.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=chosen,
            ),
        )
    trained = trainable(adapter.model, chosen)
    cfg = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("resume", "output_root", "validate_only", "baseline_run")
    }
    cfg.update(
        kind="training",
        settings=model_settings,
        target_modules=chosen,
        seed=2026,
        batch_size=1,
    )
    out = run_folder(a.output_root, cfg, m)
    save(out / "trainable.json", trained)
    meta = dict(settings=model_settings, manifest_hash=digest(m), target_modules=chosen)
    if (out / "complete.json").exists():
        print("Training already complete")
        return
    train = [s for s in m["sets"] if s["partition"] == "train"]
    val = [s for s in m["sets"] if s["partition"] == "validation"]
    if a.smoke_test:
        train = train[:2]
        val = val[:2]
    per_epoch = math.ceil(len(train) / a.gradient_accumulation)
    total = 2 if a.smoke_test else a.epochs * per_epoch
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=a.learning_rate
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=torch.cuda.is_available() and a.precision in ("4bit", "fp16")
    )
    best = float("inf")
    start = 0
    journal = out / "losses.jsonl"
    logged = rows(journal)
    if a.resume:
        # Resume only trusted checkpoints that this script wrote.
        state = torch.load(
            a.resume / "training_state.pt", map_location="cpu", weights_only=False
        )
        if state["config_hash"] != digest(cfg):
            raise ValueError("Checkpoint training settings differ")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
        best = state["best"]
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if logged and max(map(int, logged)) > start:
            raise ValueError(
                "Use the latest checkpoint, not one before recorded losses"
            )
    elif logged:
        raise ValueError("Partial run exists: use --resume checkpoint path")

    def loss(item, rotation):
        prepared = inputs(adapter, item, a.data_dir, rotation)
        label = next(l for l, r in mapping(rotation).items() if r == item["rule"])
        ids = encode(adapter, label)
        if prepared["input_ids"].shape[1] + len(ids) > a.max_tokens:
            raise ValueError(
                "Input exceeds --max-tokens. Adjust resolution explicitly; no image is dropped."
            )
        return -answer_logp(adapter.model, prepared, ids) / len(ids)

    try:
        for step in range(start, total):
            epoch, chunk = divmod(step, per_epoch)
            order = list(range(len(train)))
            random.Random(2026 + epoch).shuffle(order)
            indices = order[
                chunk * a.gradient_accumulation : (chunk + 1) * a.gradient_accumulation
            ]
            adapter.model.train()
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for i in indices:
                rotation = next(
                    r for r in range(3) if mapping(r) == train[i]["option_mapping"]
                )
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16 if a.precision == "bf16" else torch.float16,
                    enabled=torch.cuda.is_available() and a.precision != "fp32",
                ):
                    one = loss(train[i], rotation)
                if not torch.isfinite(one):
                    raise ValueError("Nonfinite loss")
                value += float(one.detach()) / len(indices)
                scaler.scale(one / len(indices)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            record = dict(
                step=step + 1, training_loss=value, learning_rate=a.learning_rate
            )
            if (step + 1) % a.validation_frequency == 0 or step + 1 == total:
                adapter.model.eval()
                with torch.no_grad():
                    record["validation_loss"] = sum(
                        float(loss(item, r)) for item in val for r in range(3)
                    ) / (len(val) * 3)
                if not math.isfinite(record["validation_loss"]):
                    raise ValueError("Nonfinite validation loss")
                if record["validation_loss"] < best:
                    best = record["validation_loss"]
                    path = out / "best_adapter"
                    adapter.model.save_pretrained(path)
                    adapter.processor.save_pretrained(path)
                    save(path / "experiment.json", meta)
                    save(
                        out / "selection.json",
                        dict(
                            step=step + 1,
                            criterion="lowest validation answer-token loss over three rotations",
                            loss=best,
                            adapter=str(path.resolve()),
                        ),
                    )
            if (step + 1) % a.checkpoint_interval == 0 or step + 1 == total:
                path = out / f"checkpoint-{step + 1}"
                adapter.model.save_pretrained(path)
                adapter.processor.save_pretrained(path)
                save(path / "experiment.json", meta)
                torch.save(
                    dict(
                        config_hash=digest(cfg),
                        step=step + 1,
                        best=best,
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                        rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else [],
                    ),
                    path / "training_state.pt",
                )
            if str(step + 1) not in logged:
                append(journal, str(step + 1), record)
            print(record, flush=True)
            memory()
    except torch.OutOfMemoryError as error:
        memory()
        raise RuntimeError(
            "CUDA memory exhausted. Lower --max-pixels or --rank, then use the same preprocessing for both comparison models. Ten images were not dropped."
        ) from error
    save(
        out / "complete.json",
        dict(steps=total, best_validation_loss=best, test_used=False),
    )


if __name__ == "__main__":
    main()
