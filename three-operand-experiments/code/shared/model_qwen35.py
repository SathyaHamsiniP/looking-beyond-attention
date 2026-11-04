"""Qwen3.5 loading and full conditional answer-sequence scoring."""

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "01-operator-controls"))
sys.path.insert(0, str(CODE / "02-probing"))
from operator_controls_qwen35 import Qwen35Adapter
from probe_concepts import decoder_layers as legacy_decoder_layers

MODEL_ID = "Qwen/Qwen3.5-4B"


def decoder_layers(model):
    # PEFT adds two wrapper levels; unwrap through its public API.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return legacy_decoder_layers(base)


def load(
    precision="4bit",
    revision="main",
    adapter_path=None,
    max_pixels=128 * 28 * 28,
    training=False,
):
    import operator_controls_qwen35 as legacy
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
    )

    legacy.MAX_PIXELS = max_pixels
    if precision == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("4-bit workflow requires a CUDA GPU")
    dtype = {
        "4bit": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]
    kw = dict(
        revision=revision,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()}
        if training and torch.cuda.is_available()
        else "auto",
    )
    if precision == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, **kw)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
    if adapter_path:
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(adapter_path)
        if config.base_model_name_or_path != MODEL_ID:
            raise ValueError("Adapter base model mismatch")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return Qwen35Adapter(model, processor, precision)


def answer_tokens(tokenizer, labels):
    result = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in labels
    }
    if any(not ids for ids in result.values()) or len(
        {tuple(x) for x in result.values()}
    ) != len(labels):
        raise ValueError("Empty or duplicate answer encodings")
    return result


def completion_inputs(inputs, tokens):
    import torch

    out = dict(inputs)
    ids = inputs["input_ids"]
    extra = torch.tensor([tokens], dtype=ids.dtype, device=ids.device)
    out["input_ids"] = torch.cat((ids, extra), dim=1)
    for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        if key in out:
            fill = 1 if key == "attention_mask" else 0
            out[key] = torch.cat((out[key], torch.full_like(extra, fill)), dim=1)
    # Recompute multimodal positions for the extended sequence.
    out.pop("position_ids", None)
    return out


def answer_log_likelihood(model, inputs, tokens):
    import torch

    if not tokens:
        raise ValueError("Empty answer")
    extended = completion_inputs(inputs, tokens[:-1])
    logits = (
        model(**extended, use_cache=False, logits_to_keep=len(tokens))
        .logits[0, -len(tokens) :]
        .float()
    )
    target = torch.tensor(tokens, device=logits.device)
    return logits.log_softmax(-1).gather(-1, target[:, None]).sum()


def score(adapter, inputs, mapping):
    import torch

    ids = answer_tokens(adapter.tokenizer, list(mapping))
    with torch.inference_mode():
        # Single-token labels use one forward pass. Multi-token labels use the full sequence probability.
        if all(len(v) == 1 for v in ids.values()):
            logits = (
                adapter.model(**inputs, use_cache=False, logits_to_keep=1)
                .logits[0, -1]
                .float()
            )
            values = torch.stack([logits[ids[label][0]] for label in mapping])
        else:
            values = torch.stack(
                [
                    answer_log_likelihood(adapter.model, inputs, ids[label])
                    for label in mapping
                ]
            )
        probs = values.softmax(0).cpu().tolist()
    import math

    if any(not math.isfinite(p) for p in probs) or not math.isclose(
        sum(probs), 1.0, abs_tol=1e-5
    ):
        raise ValueError("Nonfinite or unnormalized answer probabilities")
    return dict(zip(mapping.values(), map(float, probs)))


def model_metadata(adapter, labels):
    return dict(
        model_id=MODEL_ID,
        resolved_revision=getattr(adapter.model.config, "_commit_hash", None),
        answer_tokens=answer_tokens(adapter.tokenizer, labels),
        decoder_layers=len(decoder_layers(adapter.model)),
    )


def validate_adapter(path, manifest):
    from .run_utils import digest

    info = Path(path) / "pipeline.json"
    if not info.is_file():
        raise ValueError(
            "Adapter lacks pipeline.json; cannot verify task/model compatibility"
        )
    config = json.loads(info.read_text())
    if config["manifest_sha256"] != digest(manifest) or config["model_id"] != MODEL_ID:
        raise ValueError("Adapter task manifest or base model mismatch")
    return config
