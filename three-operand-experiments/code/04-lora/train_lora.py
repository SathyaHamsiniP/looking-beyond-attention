"""QLoRA on manifest training sets; validation loss selects the best adapter."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_loader import load_manifest, prompt, rotated
from shared.model_qwen35 import (
    answer_log_likelihood,
    answer_tokens,
    decoder_layers,
    load,
    model_metadata,
)
from shared.run_utils import (
    Journal,
    atomic_json,
    dataset_fingerprint,
    digest,
    prepare_run,
)

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj"


def select_targets(model, layers=None, suffixes=None):
    """Resolve actual decoder linear modules; never target the vision tower."""
    import torch

    blocks = decoder_layers(model)
    selected = list(range(1, len(blocks) + 1)) if layers is None else layers
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(n < 1 or n > len(blocks) for n in selected)
    ):
        raise ValueError(f"Layers are 1-based in 1..{len(blocks)}")
    names = {id(obj): name for name, obj in model.named_modules()}
    targets = []
    for number in selected:
        block = blocks[number - 1]
        prefix = names[id(block)]
        found = [
            f"{prefix}.{name}"
            for name, obj in block.named_modules()
            if name
            and isinstance(obj, torch.nn.Linear)
            and (suffixes is None or name.split(".")[-1] in suffixes)
        ]
        if not found:
            raise ValueError(
                f"No compatible linear targets at layer {number}; inspect Qwen3.5 modules"
            )
        targets += found
    return targets, selected


def check_trainable(model, targets):
    trainable = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    if not trainable or any(
        "lora_" not in name or not any(t + "." in name for t in targets)
        for name in trainable
    ):
        raise ValueError("Trainable parameters escape selected LoRA targets")
    return dict(parameters=trainable, count=sum(trainable.values()), targets=targets)


def require_baseline(path, manifest):
    config = json.loads((path / "config.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    if (
        config.get("partition") != "validation"
        or config["manifest_sha256"] != digest(manifest)
        or config.get("adapter")
        or config.get("symbolic")
        or config.get("smoke_test")
    ):
        raise ValueError(
            "Need completed untuned visual baseline on the identical manifest"
        )
    if metrics["n"] < len(
        [s for s in manifest["sets"] if s["partition"] == config["partition"]]
    ):
        raise ValueError("Baseline is incomplete")


def main(single=False):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-dir", "results-root", "baseline-run"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--layers", type=int, nargs="+", help="1-based decoder layers; default all"
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default=DEFAULT_TARGETS)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--validation-frequency", type=int, default=25)
    p.add_argument("--checkpoint-interval", type=int, default=25)
    p.add_argument(
        "--precision", choices=("4bit", "fp16", "bf16", "fp32"), default="4bit"
    )
    p.add_argument("--revision", default="main")
    p.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--max-sequence-length", type=int, default=16000)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    if single and (not a.layers or len(a.layers) != 1):
        p.error(
            "Single-layer training requires exactly one --layers number; start a new process per layer"
        )
    if (
        min(
            a.rank,
            a.alpha,
            a.batch_size,
            a.gradient_accumulation,
            a.epochs,
            a.validation_frequency,
            a.checkpoint_interval,
        )
        < 1
    ):
        p.error("Counts must be positive")
    m = load_manifest(a.manifest, a.data_dir)
    if not a.validate_only:
        require_baseline(a.baseline_run, m)
        baseline = json.loads((a.baseline_run / "config.json").read_text())
        if (
            baseline.get("revision") != a.revision
            or baseline.get("max_pixels") != a.max_pixels
        ):
            raise ValueError("Baseline model revision or image resolution differs")
    train = [s for s in m["sets"] if s["partition"] == "train"]
    val = [s for s in m["sets"] if s["partition"] == "validation"]
    if a.validate_only:
        print(
            f"Validated train={len(train)}, validation={len(val)}; final-test unused. GPU module checks remain."
        )
        return
    import numpy as np
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    adapter = load(a.precision, a.revision, max_pixels=a.max_pixels, training=True)
    targets, selected = select_targets(
        adapter.model, a.layers, a.target_modules.split(",")
    )
    for param in adapter.model.parameters():
        param.requires_grad = False
    checkpointing = not a.no_gradient_checkpointing
    if a.precision == "4bit":
        adapter.model = prepare_model_for_kbit_training(
            adapter.model,
            use_gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif checkpointing:
        adapter.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if a.resume:
        adapter.model = PeftModel.from_pretrained(
            adapter.model, a.resume, is_trainable=True
        )
    else:
        config = LoraConfig(
            r=a.rank,
            lora_alpha=a.alpha,
            lora_dropout=a.dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        adapter.model = get_peft_model(adapter.model, config)
    trained = check_trainable(adapter.model, targets)
    metadata = model_metadata(
        adapter,
        list("XYZ") if len(m["rules"]) == 3 else list("DEFGHIJK")[: len(m["rules"])],
    )
    if (
        baseline.get("model")
        and metadata["resolved_revision"] != baseline["model"]["resolved_revision"]
    ):
        raise ValueError("Resolved base revision differs from baseline; pin a commit")
    cfg = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("results_root", "resume", "validate_only")
    }
    cfg.update(
        model=metadata,
        targets=targets,
        selected_layers=selected,
        variant="single-layer" if single else "across-layers",
        data_fingerprint=dataset_fingerprint(m, a.data_dir),
    )
    out = prepare_run(a.results_root, 4, m, cfg)
    atomic_json(out / "trainable.json", trained)
    atomic_json(out / "model.json", metadata)
    if (out / "training_complete.json").exists():
        print("Training already complete:", out)
        return
    if a.smoke_test:
        train = train[:2]
        val = val[:2]
    journal = Journal(out / "losses.jsonl")
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=a.learning_rate
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=torch.cuda.is_available() and a.precision in ("4bit", "fp16")
    )
    chunk = a.batch_size * a.gradient_accumulation
    chunks_per_epoch = math.ceil(len(train) / chunk)
    total = a.max_steps or a.epochs * chunks_per_epoch
    if a.smoke_test:
        total = min(total, 2)
    start = 0
    best = float("inf")
    if a.resume:
        # Only load state files produced by this trusted local training run.
        state = torch.load(
            a.resume / "training_state.pt", map_location="cpu", weights_only=False
        )
        if state["config_hash"] != digest(cfg):
            raise ValueError("Resume configuration mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
        best = state["best"]
        torch.set_rng_state(state["torch_rng"])
        if journal.rows and max(map(int, journal.rows)) > start:
            raise ValueError(
                "Checkpoint predates recorded losses; choose the latest checkpoint"
            )
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif journal.rows:
        raise ValueError("Partial run exists; resume an explicit checkpoint")

    def loss_for(item, rotation):
        scene = rotated(item, rotation)
        paths, text, mapping = prompt(scene, m, a.data_dir)
        target = next(label for label, rule in mapping.items() if rule == item["rule"])
        inputs = adapter._inputs(paths, text)
        tokens = answer_tokens(adapter.tokenizer, [target])[target]
        if inputs["input_ids"].shape[1] + len(tokens) > a.max_sequence_length:
            raise ValueError(
                "Input exceeds token budget. Reduce --max-pixels explicitly; images are never dropped."
            )
        return -answer_log_likelihood(adapter.model, inputs, tokens) / len(tokens)

    adapter_info = dict(
        manifest_sha256=digest(m),
        model_id="Qwen/Qwen3.5-4B",
        revision=a.revision,
        resolved_revision=metadata["resolved_revision"],
        targets=targets,
        layers=selected,
    )

    def checkpoint(step):
        path = out / f"checkpoint-{step}"
        path.mkdir(exist_ok=True)
        adapter.model.save_pretrained(path)
        adapter.processor.save_pretrained(path)
        atomic_json(path / "pipeline.json", adapter_info)
        torch.save(
            dict(
                config_hash=digest(cfg),
                step=step,
                best=best,
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            ),
            path / "training_state.pt",
        )
        return path

    for step in range(start, total):
        epoch = step // chunks_per_epoch
        within = step % chunks_per_epoch
        order = list(range(len(train)))
        random.Random(a.seed + epoch).shuffle(order)
        batch = order[within * chunk : (within + 1) * chunk]
        adapter.model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.0
        for index in batch:
            rotation = random.Random(a.seed + epoch * len(train) + index).randrange(
                len(m["rules"])
            )
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16 if a.precision == "bf16" else torch.float16,
                enabled=torch.cuda.is_available() and a.precision != "fp32",
            ):
                loss = loss_for(train[index], rotation)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            training_loss += float(loss.detach()) / len(batch)
            scaler.scale(loss / len(batch)).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in adapter.model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        record = dict(step=step + 1, training_loss=training_loss)
        if (step + 1) % a.validation_frequency == 0 or step + 1 == total:
            adapter.model.eval()
            with torch.no_grad():
                record["validation_loss"] = sum(
                    float(loss_for(item, 0)) for item in val
                ) / len(val)
            if not math.isfinite(record["validation_loss"]):
                raise ValueError("Nonfinite validation loss")
            if record["validation_loss"] < best:
                best = record["validation_loss"]
                adapter.model.save_pretrained(out / "best_adapter")
                adapter.processor.save_pretrained(out / "best_adapter")
                atomic_json(out / "best_adapter/pipeline.json", adapter_info)
                atomic_json(
                    out / "selection.json",
                    dict(
                        criterion="lowest validation answer-token NLL, fixed manifest option orders",
                        step=step + 1,
                        validation_loss=best,
                        adapter=str(out / "best_adapter"),
                    ),
                )
        # Checkpoint includes optimizer, scaler, RNG; step is the next full optimizer boundary.
        if (step + 1) % a.checkpoint_interval == 0 or step + 1 == total:
            checkpoint(step + 1)
        key = str(step + 1)
        if key not in journal.rows:
            journal.add(key, record)
        print(record, flush=True)
    atomic_json(
        out / "training_complete.json",
        dict(
            steps=total,
            best_validation_loss=best,
            final_test_used=False,
            trainable_parameters=trained["count"],
        ),
    )


if __name__ == "__main__":
    main()
