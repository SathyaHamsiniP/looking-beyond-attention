"""Small helpers for this experiment. The older baseline files are read, not edited."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "01-operator-controls"))
sys.path.insert(0, str(CODE / "02-probing"))
import operator_controls_qwen35 as original
from probe_concepts import decoder_layers as find_layers
from build_training_sets import RULES

MODEL = "Qwen/Qwen3.5-4B"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def versions():
    out = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE, text=True
        ).strip()
    }
    for package in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def run_folder(root, config, manifest):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        json.dumps(
            {**config, "manifest_hash": digest(manifest), "software": versions()}
        )
    )
    out = root / digest(config)[:16]
    out.mkdir(exist_ok=True)
    if (out / "config.json").exists() and json.loads(
        (out / "config.json").read_text()
    ) != config:
        raise ValueError("Saved settings differ")
    save(out / "config.json", config)
    save(out / "manifest.json", manifest)
    print("Run folder:", out.resolve(), flush=True)
    return out


def rows(path):
    if not Path(path).exists():
        return {}
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["key"] in result:
            raise ValueError("Duplicate saved result")
        result[row["key"]] = row
    return result


def append(path, key, row):
    import os

    with Path(path).open("a") as f:
        f.write(json.dumps({**row, "key": key}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def layers(model):
    return find_layers(
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )


def mapping(rotation):
    return original.layouts("cyclic")[rotation][0]


def inputs(adapter, item, data, rotation):
    # Reuse the original image prompt, image ordering and processor adapter.
    original.IMAGE_DIR = Path(data) / "images"
    paths, text = original.build_input(
        item, mapping(rotation), original.LABELS, "images"
    )
    return adapter._inputs(paths, text)


def settings(a):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", a.revision):
        raise ValueError(
            "--revision must be the full 40-character model commit, not main"
        )
    return dict(
        model=MODEL,
        revision=a.revision,
        precision=a.precision,
        max_pixels=a.max_pixels,
        prompt_hash=sha_file(CODE / "01-operator-controls/operator_controls_qwen35.py"),
        scoring="conditional answer-sequence log likelihood, normalized over XYZ",
    )


def load_model(a, training=False):
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
    )

    settings(a)
    if a.precision == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("4-bit loading needs a CUDA GPU")
    original.MAX_PIXELS = a.max_pixels
    dtype = {
        "4bit": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[a.precision]
    kwargs = dict(
        revision=a.revision,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": 0} if training and torch.cuda.is_available() else "auto",
    )
    if a.precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, **kwargs)
    processor = AutoProcessor.from_pretrained(MODEL, revision=a.revision)
    adapter_path = getattr(a, "adapter", None)
    if adapter_path:
        from peft import PeftModel

        meta = json.loads((Path(adapter_path) / "experiment.json").read_text())
        if meta["settings"] != settings(a):
            raise ValueError("Adapter model revision or preprocessing differs")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return original.Qwen35Adapter(model, processor, a.precision)


def encode(adapter, label):
    ids = adapter.tokenizer.encode(label, add_special_tokens=False)
    if not ids:
        raise ValueError("Empty answer tokenization")
    return ids


def answer_logp(model, prepared, ids):
    import torch

    extended = dict(prepared)
    tail = torch.tensor(
        [ids[:-1]],
        device=prepared["input_ids"].device,
        dtype=prepared["input_ids"].dtype,
    )
    extended["input_ids"] = torch.cat([prepared["input_ids"], tail], 1)
    for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        if key in extended:
            extended[key] = torch.cat(
                [
                    extended[key],
                    torch.full_like(tail, 1 if key == "attention_mask" else 0),
                ],
                1,
            )
    extended.pop("position_ids", None)
    logits = (
        model(**extended, use_cache=False, logits_to_keep=len(ids))
        .logits[0, -len(ids) :]
        .float()
    )
    target = torch.tensor(ids, device=logits.device)
    # The last prompt token predicts the first answer token. Prompt tokens have no loss.
    return logits.log_softmax(-1).gather(1, target[:, None]).sum()


def score(adapter, prepared, rotation):
    import torch

    with torch.inference_mode():
        ids = {label: encode(adapter, label) for label in "XYZ"}
        if len({tuple(v) for v in ids.values()}) != 3:
            raise ValueError("Answer labels share token IDs")
        if all(len(v) == 1 for v in ids.values()):
            logits = (
                adapter.model(**prepared, use_cache=False, logits_to_keep=1)
                .logits[0, -1]
                .float()
            )
            values = torch.stack([logits[ids[l][0]] for l in "XYZ"])
        else:
            values = torch.stack(
                [answer_logp(adapter.model, prepared, ids[l]) for l in "XYZ"]
            )
        probs = values.softmax(0).cpu().tolist()
        if not all(__import__("math").isfinite(p) for p in probs):
            raise ValueError("Nonfinite answer scores")
    return {mapping(rotation)[l]: float(p) for l, p in zip("XYZ", probs)}


def adapter_identity(path):
    return (
        None
        if not path
        else {p.name: sha_file(p) for p in sorted(Path(path).iterdir()) if p.is_file()}
    )


def model_args(p):
    p.add_argument("--revision", required=True, help="Full model commit SHA")
    p.add_argument(
        "--precision", choices=("4bit", "fp16", "bf16", "fp32"), default="4bit"
    )
    p.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")


def memory():
    import torch

    if torch.cuda.is_available():
        print(
            f"CUDA allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )


def matched(left, right, fields):
    for field in fields:
        if left[field] != right[field]:
            raise ValueError("Cannot compare different " + field)
