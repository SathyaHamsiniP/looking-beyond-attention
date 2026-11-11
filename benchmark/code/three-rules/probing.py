"""Experiment 02: matched 1-diagnostic vs 10-image, frozen-Qwen layer probes.

No imports from historical experiments. Source datasets/rules are read from manifests,
so later benchmark splits can be regenerated without rewriting probe extraction.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import REPO, source_path

import re

import numpy as np

from build_sets import compatible_rules, read_json, read_rules, sha256_file
from operator_controls import MODEL, REVISION, original_prompt
from pilot_10_vs_20 import evaluate, load_model, mapping

CONDITIONS = (1, 10)
SEED = 2026


def select_diagnostic(item, rules):
    """Return the first scene that uniquely supports the generating rule."""
    unique = []
    for position, (name, bits) in enumerate(zip(item["image_files"], item["bits"])):
        if compatible_rules([bits], rules) == [item["rule"]]:
            unique.append({"filename": name, "bits": bits, "position": position + 1})
    if not unique:
        raise ValueError(f"No single-image diagnostic witness in {item['set_id']}")
    return unique[0]


def validate(data_dir, manifest_path, rules_path, train_split, test_split):
    spec = read_rules(rules_path)
    rules = spec["rules"]
    m = read_json(manifest_path)
    if len(rules) != 3 or m["rule_ids"] != [r["id"] for r in rules]:
        raise ValueError("This probe variation currently supports three rule choices")
    if m["images_per_set"] != 10:
        raise ValueError("Build a ten-image base manifest; the 1-image condition is derived")
    if m["source_metadata_sha256"] != sha256_file(Path(data_dir) / "metadata.json"):
        raise ValueError("Dataset metadata has changed")
    if m["rules_sha256"] != sha256_file(rules_path):
        raise ValueError("Manifest uses a different rules.json")
    if train_split == test_split:
        raise ValueError("Train and evaluation split names must differ")
    records = read_json(Path(data_dir) / "metadata.json")["records"]
    images = {r["filename"]: r for r in records}
    if len(images) != len(records):
        raise ValueError("Duplicate filenames in dataset metadata")
    # The manifest stores a split for each set; validate against the source metadata.
    # External partition maps are allowed only when explicitly supplied in the manifest.
    partitions = m.get("image_partition_counts", {})
    if not {train_split, test_split} <= set(partitions):
        raise ValueError("Both requested partitions must exist in the source manifest")
    seen_ids = set()
    files_by_split = defaultdict(set)
    counts = Counter()
    chosen = {}
    for item in m["sets"]:
        sid, split, rule = item["set_id"], item["split"], item["rule"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sid) or sid in seen_ids:
            raise ValueError("Set identifiers must be unique and safe for filenames")
        seen_ids.add(sid)
        if split not in (train_split, test_split):
            continue
        if rule not in m["rule_ids"] or len(item["image_files"]) != 10:
            raise ValueError(f"Invalid set: {sid}")
        if len(item["bits"]) != 10 or len(set(item["image_files"])) != 10:
            raise ValueError(f"Duplicate or missing image in set: {sid}")
        for name, bit in zip(item["image_files"], item["bits"]):
            if name not in images or not (Path(data_dir) / "images" / name).is_file():
                raise ValueError(f"Missing image {name}")
            if "".join(map(str, images[name]["bits"])) != bit:
                raise ValueError(f"Image labels differ: {name}")
            if m["partitions_sha256"] is None and images[name]["split"] != split:
                raise ValueError(f"Image {name} crosses metadata split")
            files_by_split[split].add(name)
        if compatible_rules(item["bits"], rules) != [rule]:
            raise ValueError(f"Not uniquely identifiable: {sid}")
        chosen[sid] = select_diagnostic(item, rules)
        counts[(split, rule)] += 1
    if not files_by_split[train_split].isdisjoint(files_by_split[test_split]):
        raise ValueError("Train/test image leakage detected")
    for split in (train_split, test_split):
        for rule in m["rule_ids"]:
            if counts[(split, rule)] < 3:
                raise ValueError(f"Need at least 3 sets per rule in {split}")
    # Repetition within a partition is permitted but summarized as a limitation.
    summary = {
        "sets_by_split_and_rule": {
            split: {rule: counts[(split, rule)] for rule in m["rule_ids"]}
            for split in (train_split, test_split)
        },
        "diagnostic_unique_files": {
            split: len({chosen[s["set_id"]]["filename"] for s in m["sets"]
                        if s["split"] == split})
            for split in (train_split, test_split)
        },
        "diagnostic_total_sets": {
            split: sum(1 for s in m["sets"] if s["split"] == split)
            for split in (train_split, test_split)
        },
        "train_test_image_disjoint": True,
    }
    selected = [s for s in m["sets"] if s["split"] in (train_split, test_split)]
    return rules, selected, chosen, summary


def decoder_layers(model):
    """Qwen3.5 decoder hooks; avoid assumptions about attention implementation."""
    for path in ("model.language_model", "model.language_model.model",
                 "model.text_model", "model", "language_model"):
        obj = model
        for name in path.split("."):
            obj = getattr(obj, name, None)
        if hasattr(obj, "layers"):
            return obj.layers
    raise ValueError("Cannot find Qwen decoder layers; check Transformers version")


def capture(model, processor, token_ids, files, data_dir, prompt, max_pixels):
    """Call the SAME model-input/scoring helper as operator controls, with hooks."""
    import torch
    layers = decoder_layers(model)
    collected = {}
    handles = []

    def hook(i):
        def take(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.ndim != 3:
                raise ValueError(f"Unexpected decoder output at layer {i}")
            # Immediately pool the final token; don't retain the full visual sequence.
            collected[i] = h[0, -1, :].detach().float().cpu().numpy().copy()
        return take

    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(hook(i)))
        probabilities, tokens = evaluate(
            model, processor, token_ids, files, data_dir, prompt, max_pixels)
    finally:
        for handle in handles:
            handle.remove()
    if len(collected) != len(layers):
        raise ValueError("Some decoder hooks did not run")
    feature = np.stack([collected[i] for i in range(len(layers))])
    if feature.ndim != 2 or not np.isfinite(feature).all():
        raise ValueError("Invalid probe activation array")
    return feature, probabilities, tokens


def feature_path(out, set_id, condition):
    return out / "features" / f"{set_id}_n{condition}.npz"


def save_features(path, activations, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, features=activations, input_tokens=tokens)
    tmp.replace(path)


def load_features(path):
    with np.load(path) as arr:
        value = arr["features"]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"Invalid saved feature: {path}")
    return value


def fit_probes(out, items, row_map, train_split, test_split, rules):
    """Train one frozen-feature linear probe per layer and condition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    names = [r["id"] for r in rules]
    train = [s for s in items if s["split"] == train_split]
    test = [s for s in items if s["split"] == test_split]
    y_train = np.array([s["rule"] for s in train])
    y_test = np.array([s["rule"] for s in test])
    if set(y_train) != set(names) or set(y_test) != set(names):
        raise ValueError("Both splits must contain all rule classes")
    all_predictions = []
    accuracies = []
    all_results = {}
    for condition in CONDITIONS:
        train_data = np.stack(
            [load_features(feature_path(out, s["set_id"], condition)) for s in train])
        test_data = np.stack(
            [load_features(feature_path(out, s["set_id"], condition)) for s in test])
        if train_data.ndim != 3 or test_data.shape[1:] != train_data.shape[1:]:
            raise ValueError("Feature shapes disagree across train and test")
        num_layers = train_data.shape[1]
        for layer in range(num_layers):
            for control in ("true", "shuffled_train_labels"):
                # Same permutation per layer in both conditions, so controls are matched.
                label_train = y_train.copy()
                if control == "shuffled_train_labels":
                    np.random.default_rng(SEED + layer).shuffle(label_train)
                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
                probe.fit(train_data[:, layer, :], label_train)
                prediction = probe.predict(test_data[:, layer, :])
                probs = probe.predict_proba(test_data[:, layer, :])
                classes = list(probe.classes_)
                accuracy = float(np.mean(prediction == y_test))
                accuracies.append({
                    "images": condition, "layer": layer + 1,
                    "control": control, "train_sets": len(train),
                    "test_sets": len(test), "accuracy": accuracy,
                })
                for i, item in enumerate(test):
                    all_predictions.append({
                        "images": condition, "layer": layer + 1, "control": control,
                        "set_id": item["set_id"], "true_rule": item["rule"],
                        "predicted_rule": str(prediction[i]),
                        "correct": int(prediction[i] == y_test[i]),
                        "true_probability": float(probs[i, classes.index(y_test[i])])
                        if y_test[i] in classes else 0.0,
                    })
        answer_rows = [row_map[(s["set_id"], condition)] for s in test]
        all_results[str(condition)] = {
            "qwen_fixed_rotation_accuracy": sum(r["correct"] for r in answer_rows) / len(answer_rows),
            "qwen_predictions": dict(Counter(r["predicted_rule"] for r in answer_rows)),
            "probe_final_layer_accuracy": next(
                x["accuracy"] for x in accuracies
                if x["images"] == condition and x["layer"] == num_layers
                and x["control"] == "true"),
            "probe_final_layer_shuffled_control_accuracy": next(
                x["accuracy"] for x in accuracies
                if x["images"] == condition and x["layer"] == num_layers
                and x["control"] == "shuffled_train_labels"),
        }

    with (out / "layer_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(accuracies[0]))
        writer.writeheader()
        writer.writerows(accuracies)
    with (out / "probe_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    summary = {
        "conditions": list(CONDITIONS),
        "rule_ids": names,
        "train_sets": len(train),
        "test_sets": len(test),
        "chance": 1 / len(names),
        "layers": num_layers,
        "evaluation": all_results,
        "reading": ("Inspect all layers in layer_accuracy.csv. Do not select the "
                    "best layer on test and report it as tuned held-out accuracy."),
        "limitation": ("The 1-image diagnostic rule can be identified from one "
                       "visible Boolean scene configuration; strong decoding "
                       "could reflect concept recognition, not rule reasoning. "
                       "Only a fixed answer-option rotation is evaluated. "
                       "Images can repeat within each partition, but never "
                       "across train/test partitions."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True,
                   help="10-image sets from benchmark/build_sets.py")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--rules", type=Path, default=source_path("rules.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--fit-only", action="store_true",
                   help="Train probes from already extracted features; CPU is enough")
    a = p.parse_args()
    rules, items, witnesses, split_info = validate(
        a.data_dir, a.manifest, a.rules, a.train_split, a.test_split)
    print("Paired probing data validated:", json.dumps(split_info), flush=True)
    if a.validate_only:
        return
    if a.smoke_test and a.fit_only:
        p.error("Choose either --smoke-test or --fit-only")

    cfg = {
        "task": "02-probing-1-vs-10-v1", "model": MODEL, "revision": REVISION,
        "quantization": "4bit-NF4", "manifest_sha256": sha256_file(a.manifest),
        "rules_sha256": sha256_file(a.rules),
        "data_sha256": sha256_file(a.data_dir / "metadata.json"),
        "max_pixels": a.max_pixels, "train_split": a.train_split,
        "test_split": a.test_split, "conditions": list(CONDITIONS),
        "feature": "decoder final-token activation after every layer",
        "readout": "next token restricted X/Y/Z softmax; thinking=False",
        "rotation": 0, "probe": "standardize + multinomial logistic regression C=1",
        "seed": SEED,
    }
    if a.smoke_test:
        work = [next(item for item in items if item["split"] == a.train_split)]
        rows, row_map = [], {}
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        cfg_path = a.out / "config.json"
        if cfg_path.exists() and read_json(cfg_path) != cfg:
            raise ValueError("Existing output uses different experiment settings")
        cfg_path.write_text(json.dumps(cfg, indent=2)+"\n")
        log = a.out / "model_predictions.jsonl"
        rows = ([json.loads(x) for x in log.read_text().splitlines() if x.strip()]
                if log.exists() else [])
        row_map = {(r["set_id"], r["images"]): r for r in rows}
        if len(row_map) != len(rows):
            raise ValueError("Duplicate model-prediction keys")
        work = items

    expected = {(s["set_id"], n) for s in work for n in CONDITIONS}
    if not set(row_map) <= expected:
        raise ValueError("Unexpected stored prediction keys")
    jobs = [(s, n) for s in work for n in CONDITIONS]
    missing = [(s, n) for s, n in jobs
               if ((s["set_id"], n) not in row_map or
                   not feature_path(a.out, s["set_id"], n).exists())]
    if not a.fit_only and missing:
        model, processor, ids = load_model()
        for item, n in missing:
            sid = item["set_id"]
            filename_list = ([witnesses[sid]["filename"]] if n == 1
                             else item["image_files"])
            prompt = original_prompt(rules, mapping([r["id"] for r in rules], 0), n)
            feat, probs, tokens = capture(model, processor, ids, filename_list,
                                          a.data_dir, prompt, a.max_pixels)
            label_map = mapping([r["id"] for r in rules], 0)
            label = max("XYZ", key=lambda x: probs[x])
            true_label = next(x for x in "XYZ" if label_map[x] == item["rule"])
            row = {
                "set_id": sid, "split": item["split"], "images": n,
                "true_rule": item["rule"], "predicted_label": label,
                "predicted_rule": label_map[label], "probabilities": probs,
                "true_label": true_label, "correct": label_map[label] == item["rule"],
                "input_tokens": tokens, "image_files": filename_list,
            }
            print(f"{sid} n={n}: Qwen predicts {label_map[label]}; "
                  f"features={tuple(feat.shape)}", flush=True)
            if a.smoke_test:
                continue
            save_features(feature_path(a.out, sid, n), feat, tokens)
            # Append only after features are safely saved.
            if (sid, n) in row_map:
                if row != row_map[(sid, n)]:
                    raise ValueError("Existing model row disagrees with rerun")
            else:
                with log.open("a") as f:
                    f.write(json.dumps(row)+"\n")
                row_map[(sid, n)] = row
    if a.smoke_test:
        print("Probe extraction smoke test passed; no files saved.")
        return
    if not all(feature_path(a.out, s["set_id"], n).is_file()
               and (s["set_id"], n) in row_map for s, n in jobs):
        raise ValueError("Missing features/predictions. Extract them before --fit-only.")
    result = fit_probes(a.out, items, row_map, a.train_split, a.test_split, rules)
    result["data_integrity"] = split_info
    (a.out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
