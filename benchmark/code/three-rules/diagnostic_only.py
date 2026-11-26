"""Experiment 01 variation: classify a rule from its single diagnostic image.

Uses the same 24 image sets, cyclic option rotations, Qwen revision and readout
as benchmark/operator_controls.py. Reads completed 10-image results for paired
comparison; never changes them or imports the historical experiment files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import REPO, source_path

import random

from build_sets import read_rules
from operator_controls import (
    MODEL, REVISION, RULES, entropy, mi, original_prompt, satisfies
)
from pilot_10_vs_20 import evaluate, load_model, mapping


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic(item):
    """Choose first diagnostic image in the original 10-image order."""
    rule = item["true_rule"]
    images = item["10"]
    bits = item["bits10"]
    if len(images) != 10 or len(bits) != 10 or len(set(images)) != 10:
        raise ValueError("Expected 10 distinct images with matching scene descriptions")
    indices = [i for i, values in enumerate(bits) if sum(values) == 1]
    correct = [i for i in indices if satisfies(rule, bits[i]) and
               all(not satisfies(other, bits[i]) for other in RULES if other != rule)]
    if not correct:
        raise ValueError(f"Missing rule-distinguishing image in {item['group_id']}")
    i = correct[0]
    return {"filename": images[i], "bits": bits[i], "position_in_ten": i + 1,
            "diagnostic_count_in_ten": len(correct)}


def load_reference(ref_dir, data_dir, rules_path):
    ref_dir, data_dir = Path(ref_dir), Path(data_dir)
    config = json.loads((ref_dir / "config.json").read_text())
    sets = json.loads((ref_dir / "paired_sets.json").read_text())
    rules = read_rules(rules_path)["rules"]
    if [r["id"] for r in rules] != list(RULES):
        raise ValueError("Expected original R1/R2/R3 rule definitions")
    expected = {
        "protocol": "original-main_72-coverage-cyclic-v1",
        "model": MODEL, "revision": REVISION,
        "quantization": "4bit-NF4", "image_counts": [10, 20],
        "rotations": 3, "rules_sha256": sha(rules_path),
        "data_sha256": sha(data_dir / "metadata.json"),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Reference differs in {key}: cannot make a matched comparison")
    if len(sets) != 3 * config["sets_per_rule"]:
        raise ValueError("Wrong number of source sets")
    rows = [json.loads(line) for line in (ref_dir / "runs.jsonl").read_text().splitlines()
            if line.strip()]
    if len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate reference trial ID")
    reference = {r["trial_id"]: r for r in rows if r["images_per_set"] == 10}
    expected_ids = {f"{item['group_id']}:cyclic:{rotation}:images:10"
                    for item in sets.values() for rotation in range(3)}
    if set(reference) != expected_ids:
        raise ValueError("Reference 10-image evaluations incomplete or unexpected")
    choices = {}
    for key, item in sets.items():
        if item["group_id"] != key or item["true_rule"] not in RULES:
            raise ValueError("Malformed source image set")
        selected = diagnostic(item)
        if not (data_dir / "images" / selected["filename"]).is_file():
            raise ValueError(f"Selected image missing: {selected['filename']}")
        choices[key] = selected
        for rotation in range(3):
            row = reference[f"{key}:cyclic:{rotation}:images:10"]
            labels = mapping(list(RULES), rotation)
            if (row["group_id"] != key or row["true_rule"] != item["true_rule"]
                    or row["rotation"] != rotation or row["image_files"] != item["10"]
                    or row["label_mapping"] != labels
                    or row["prompt"] != original_prompt(rules, labels, 10)):
                raise ValueError(f"Reference mismatch for {key}, rotation {rotation}")
    return config, sets, rules, reference, choices


def metrics(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    truth = [r["true_rule"] for r in rows]
    predicted = [r["predicted_rule"] for r in rows]
    return {
        "trials": len(rows), "sets": len(groups),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows)/len(rows),
        "per_rule_accuracy": {
            rule: sum(r["correct"] for r in rows if r["true_rule"] == rule)
                  / sum(r["true_rule"] == rule for r in rows)
            for rule in RULES
        },
        "predicted_rules": dict(Counter(predicted)),
        "answer_labels": dict(Counter(r["predicted_label"] for r in rows)),
        "mutual_information_bits": mi(truth, predicted),
        "output_entropy_bits": entropy(predicted),
        "rotation_consistency": sum(
            len({r["predicted_rule"] for r in group}) == 1
            for group in groups.values()) / len(groups),
        "mean_true_probability": sum(
            r["probabilities"][r["true_label"]] for r in rows)/len(rows),
    }


def report(reference, new_rows):
    old = {r["trial_id"]: r for r in reference.values()}
    new = {r["reference_trial_id"]: r for r in new_rows}
    if set(old) != set(new):
        raise ValueError("Missing or unexpected new diagnostic-only trials")
    keys = sorted(old)
    before = [old[k] for k in keys]
    after = [new[k] for k in keys]
    # Bootstrap by image set so repeated label rotations are not independent.
    group_ids = sorted(set(r["group_id"] for r in before))
    delta_per_group = {
        g: sum(new[k]["correct"] - old[k]["correct"] for k in keys
               if old[k]["group_id"] == g)/3
        for g in group_ids
    }
    rng = random.Random(2026)
    diffs = sorted(sum(delta_per_group[g] for g in
                       rng.choices(group_ids, k=len(group_ids)))/len(group_ids)
                   for _ in range(2000))
    return {
        "diagnostic_only": metrics(after),
        "original_10_images": metrics(before),
        "paired_comparison": {
            "accuracy_delta_diagnostic_minus_ten": (
                metrics(after)["accuracy"] - metrics(before)["accuracy"]),
            "accuracy_delta_95ci_cluster": [diffs[49], diffs[1949]],
            "changed_predictions": sum(new[k]["predicted_rule"] != old[k]["predicted_rule"]
                                       for k in keys),
            "incorrect_to_correct": sum(not old[k]["correct"] and new[k]["correct"]
                                       for k in keys),
            "correct_to_incorrect": sum(old[k]["correct"] and not new[k]["correct"]
                                       for k in keys),
        },
        "interpretation_note": (
            "One selected positive image is logically sufficient to identify the "
            "rule among R1/R2/R3. Compare the same scenes/labels and model readout. "
            "The input length and the prompt's image count also change. Rotations "
            "are repeated measures, not independent examples."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True,
                   help="Completed output folder from benchmark/operator_controls.py")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="New results folder (never the reference folder itself)")
    p.add_argument("--rules", type=Path, default=source_path("rules.json"))
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    if a.out.resolve() == a.reference_dir.resolve():
        p.error("--out must be different from --reference-dir")
    cfg, sets, rules, reference, selected = load_reference(
        a.reference_dir, a.data_dir, a.rules)
    print(f"Matched diagnostic-only variation: {len(sets)} sets × 3 rotations "
          f"= {len(reference)} trials.", flush=True)
    print("Distinct diagnostic files:", len({s["filename"] for s in selected.values()}))
    if a.validate_only:
        print("All selected images uniquely identify their true rule. No GPU used.")
        return
    settings = {
        "protocol": "original-main_72-diagnostic-only-v1",
        "reference_config_sha256": sha(a.reference_dir / "config.json"),
        "reference_sets_sha256": sha(a.reference_dir / "paired_sets.json"),
        "reference_runs_sha256": sha(a.reference_dir / "runs.jsonl"),
        "rules_sha256": sha(a.rules), "model": MODEL, "revision": REVISION,
        "quantization": "4bit-NF4", "max_pixels": cfg["max_pixels"],
        "selection": "first uniquely diagnostic image in original 10-image order",
        "prompt": "original image prompt with n=1",
        "readout": "next-token X/Y/Z softmax; thinking=False",
    }
    if not a.smoke_test:
        a.out.mkdir(parents=True, exist_ok=True)
        config_path = a.out / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != settings:
            raise ValueError("Existing diagnostic-only output uses different settings")
        config_path.write_text(json.dumps(settings, indent=2)+"\n")
        choices_path = a.out / "selected_images.json"
        if choices_path.exists() and json.loads(choices_path.read_text()) != selected:
            raise ValueError("The selected diagnostic images have changed")
        choices_path.write_text(json.dumps(selected, indent=2)+"\n")
        out_path = a.out / "runs.jsonl"
        rows = ([json.loads(line) for line in out_path.read_text().splitlines()
                 if line.strip()] if out_path.exists() else [])
        keys = [r["trial_id"] for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate saved diagnostic-only trials")
        done = set(keys)
    else:
        out_path, rows, done = None, [], set()
    jobs = [(f"{key}:cyclic:{rotation}:images:1", key, rotation, item)
            for key, item in sets.items() for rotation in range(3)]
    if not done <= {job[0] for job in jobs}:
        raise ValueError("Saved diagnostic-only trials do not match these inputs")
    if a.smoke_test:
        jobs = jobs[:1]
    incomplete = [job for job in jobs if job[0] not in done]
    if incomplete:
        model, processor, token_ids = load_model()
        for job_id, key, rotation, item in incomplete:
            labels = mapping(list(RULES), rotation)
            prompt = original_prompt(rules, labels, 1)
            image_name = selected[key]["filename"]
            probabilities, token_count = evaluate(
                model, processor, token_ids, [image_name],
                a.data_dir, prompt, cfg["max_pixels"])
            label = max("XYZ", key=lambda l: probabilities[l])
            true_label = next(l for l in "XYZ" if labels[l] == item["true_rule"])
            row = {
                "trial_id": job_id,
                "reference_trial_id": f"{key}:cyclic:{rotation}:images:10",
                "group_id": key, "true_rule": item["true_rule"],
                "rotation": rotation, "images_per_set": 1,
                "image_files": [image_name], "scene_bits": [selected[key]["bits"]],
                "diagnostic_position_in_original_ten": selected[key]["position_in_ten"],
                "label_mapping": labels, "prompt": prompt,
                "probabilities": probabilities,
                "predicted_label": label, "predicted_rule": labels[label],
                "true_label": true_label,
                "correct": labels[label] == item["true_rule"],
                "input_tokens": token_count,
            }
            print(f"{len(rows)+1}/{len(jobs)} {job_id}: "
                  f"truth={item['true_rule']} prediction={labels[label]} "
                  f"{'OK' if row['correct'] else 'wrong'}", flush=True)
            if a.smoke_test:
                print("Diagnostic-only smoke test passed; no results saved.")
                return
            with out_path.open("a") as f:
                f.write(json.dumps(row)+"\n")
            rows.append(row)
    if not a.smoke_test:
        if len(rows) != len(jobs):
            raise ValueError("Missing evaluations; rerun to resume")
        result = report(reference, rows)
        (a.out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
