"""Paired pretrained 10-vs-20 image pilot. No imports from old experiments."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import REPO, source_path

import time

from build_sets import read_json, read_rules


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LABELS = "XYZ"


def pairs(manifest_10, manifest_20, data_dir, rules_path):
    a, b = read_json(manifest_10), read_json(manifest_20)
    spec = read_rules(rules_path)
    if a["images_per_set"] != 10 or b["images_per_set"] != 20:
        raise ValueError("Expected 10- and 20-image manifests")
    for key in ("seed", "rule_ids", "operand_names", "source_metadata_sha256",
                "rules_sha256", "partitions_sha256", "sets_per_rule_by_partition"):
        if a[key] != b[key]:
            raise ValueError(f"Manifest mismatch: {key}")
    if a["rule_ids"] != [r["id"] for r in spec["rules"]]:
        raise ValueError("Manifest rules differ from rules.json")
    if len(a["rule_ids"]) != 3:
        raise ValueError("This small pilot runner currently supports three rules only")
    x, y = a["sets"], b["sets"]
    if len(x) != len(y) or not x:
        raise ValueError("Different or empty set counts")
    meta_hash = hashlib.sha256((Path(data_dir) / "metadata.json").read_bytes()).hexdigest()
    if meta_hash != a["source_metadata_sha256"]:
        raise ValueError("Dataset metadata differs from the source used for manifests")
    folder = Path(data_dir) / "images"
    for left, right in zip(x, y):
        for key in ("set_id", "split", "rule"):
            if left[key] != right[key]:
                raise ValueError(f"Different paired set {key}")
        if right["image_files"][:10] != left["image_files"]:
            raise ValueError("Twenty-image set must start with exact ten-image set")
        if any(not (folder / name).is_file() for name in right["image_files"]):
            raise ValueError(f"Missing image for {right['set_id']}")
    if not all(s["split"] in ("validation", "test") for s in x):
        raise ValueError("Pilot must use held-out validation/test partitions")
    return list(zip(x, y)), spec


def mapping(ids, rotation):
    ordered = ids[rotation:] + ids[:rotation]
    return dict(zip(LABELS, ordered))


def prompt_for(rules, operand_names, mapping_now, n):
    objects = {"A": "green cube", "B": "blue sphere", "C": "red cylinder"}
    definitions = "\n".join(
        f"{name} is true when a {objects.get(name, name)} is present."
        for name in operand_names
    )
    by_id = {r["id"]: r["expression"] for r in rules}
    options = "\n".join(f"{l}: {by_id[mapping_now[l]]}" for l in LABELS)
    return (f"Images 1-{n} all satisfy ONE hidden Boolean rule.\n\n"
            f"{definitions}\n"
            "Each concept requires the exact colour and shape.\n"
            "Other objects are irrelevant distractors.\n\n"
            f"Candidate rules:\n{options}\n\n"
            "AND requires both inputs to be true.\n"
            "OR requires at least one input to be true, including both.\n"
            "Follow the parentheses.\n\n"
            "Pick the rule consistent with every image.\n"
            "Reply with nothing but its label: X, Y, or Z.\nANSWER:")


def load_model():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 4-bit pilot")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16,
        quantization_config=q, attn_implementation="sdpa",
        device_map="auto").eval()
    ids = [processor.tokenizer.encode(l, add_special_tokens=False) for l in LABELS]
    if any(len(v) != 1 for v in ids) or len(set(v[0] for v in ids)) != 3:
        raise ValueError("X/Y/Z must be distinct single tokens")
    return model, processor, [v[0] for v in ids]


def evaluate(model, processor, token_ids, filenames, data_dir, prompt, max_pixels):
    import torch
    from PIL import Image
    content = []
    for name in filenames:
        with Image.open(Path(data_dir) / "images" / name) as f:
            picture = f.convert("RGB")
        if picture.width * picture.height > max_pixels:
            scale = (max_pixels / (picture.width * picture.height)) ** 0.5
            picture = picture.resize((max(1, int(picture.width * scale)),
                                      max(1, int(picture.height * scale))),
                                     Image.Resampling.LANCZOS)
        content.append({"type": "image", "image": picture})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    prepared = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt").to(model.device)
    tokens = int(prepared["input_ids"].shape[1])
    with torch.inference_mode():
        logits = model(**prepared, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        scores = torch.stack([logits[x] for x in token_ids])
        result = scores.softmax(dim=0).cpu().tolist()
    return dict(zip(LABELS, result)), tokens


def summary(rows):
    if not rows:
        return {}
    answer = {}
    for size in (10, 20):
        sub = [r for r in rows if r["images_per_set"] == size]
        if not sub:
            continue
        answer[str(size)] = {
            "trials": len(sub),
            "distinct_image_sets": len(set(r["set_id"] for r in sub)),
            "accuracy": sum(r["correct"] for r in sub) / len(sub),
            "answer_labels": dict(Counter(r["predicted_label"] for r in sub)),
            "per_rule": {
                rule: sum(r["correct"] for r in sub if r["true_rule"] == rule)
                / sum(r["true_rule"] == rule for r in sub)
                for rule in sorted(set(r["true_rule"] for r in sub))
            },
        }
    answer["caution"] = ("Small pilot; option rotations are repeated evaluations of "
                         "the same sets, not independent observations. This pilot "
                         "uses the held-out 'test' partition for diagnostics only; "
                         "do not treat it as an untouched final benchmark test.")
    return answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-10", type=Path, required=True)
    p.add_argument("--manifest-20", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--rules", type=Path, default=source_path("rules.json"))
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory, ideally on Google Drive")
    p.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="One 20-image forward pass only, with no results written")
    args = p.parse_args()
    ps, spec = pairs(args.manifest_10, args.manifest_20, args.data_dir, args.rules)
    print(f"Paired manifests OK: {len(ps)} sets, 10 and 20 images, "
          f"{len(spec['rules'])} rules.", flush=True)
    if args.validate_only:
        return
    if args.max_pixels < 1024:
        p.error("--max-pixels too small")
    if args.smoke_test:
        ps = ps[:1]
    config = {"model": MODEL, "revision": REVISION, "quantization": "NF4 4-bit",
              "manifest_10_sha256": hashlib.sha256(args.manifest_10.read_bytes()).hexdigest(),
              "manifest_20_sha256": hashlib.sha256(args.manifest_20.read_bytes()).hexdigest(),
              "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest(),
              "max_pixels": args.max_pixels,
              "readout": "next-token X/Y/Z softmax",
              "prompt_version": "pilot-v1"}
    if not args.smoke_test:
        args.out.mkdir(parents=True, exist_ok=True)
        config_path = args.out / "config.json"
        if config_path.is_file() and read_json(config_path) != config:
            raise ValueError("Output directory already has a different configuration")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        path = args.out / "predictions.jsonl"
        completed = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        keys = [r["key"] for r in completed]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate completed result keys")
        done = set(keys)
    else:
        completed, done, path = [], set(), None
    required = [(left, right, rotation, size)
                for left, right in ps for rotation in range(3) for size in (10, 20)
                if not args.smoke_test or (rotation == 0 and size == 20)]
    if not args.smoke_test:
        expected_keys = {f"{left['set_id']}:{rotation}:{size}"
                         for left, right, rotation, size in required}
        if not done <= expected_keys:
            raise ValueError("Saved result keys don't match this pilot")
    if all(f"{left['set_id']}:{rotation}:{size}" in done for left, right, rotation, size in required):
        print("All trials already present. No model load needed.")
    else:
        model, processor, token_ids = load_model()
        import torch
        for left, right, rotation, size in required:
            key = f"{left['set_id']}:{rotation}:{size}"
            if key in done:
                continue
            item = left if size == 10 else right
            current = mapping([r["id"] for r in spec["rules"]], rotation)
            prompt = prompt_for(spec["rules"], spec["operand_names"], current, size)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            try:
                probs, n_tokens = evaluate(model, processor, token_ids,
                                           item["image_files"], args.data_dir,
                                           prompt, args.max_pixels)
            except torch.cuda.OutOfMemoryError:
                print(f"GPU out of memory at {size} images. Try lower --max-pixels.",
                      flush=True)
                raise
            label = max(LABELS, key=lambda lab: probs[lab])
            row = {"key": key, "set_id": item["set_id"], "true_rule": item["rule"],
                   "images_per_set": size, "rotation": rotation,
                   "option_mapping": current, "predicted_label": label,
                   "predicted_rule": current[label],
                   "probabilities": probs, "correct": current[label] == item["rule"],
                   "input_tokens": n_tokens, "seconds": round(time.monotonic()-start, 2),
                   "peak_gpu_gib": round(torch.cuda.max_memory_allocated()/2**30, 3)}
            print(f"{key} truth={item['rule']} predicted={current[label]} "
                  f"tokens={n_tokens} peak={row['peak_gpu_gib']} GiB", flush=True)
            if args.smoke_test:
                print("GPU smoke test passed; no predictions saved.")
                return
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")
            completed.append(row)
            done.add(key)
    if not args.smoke_test:
        result = summary(completed)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
