"""Original main_72 operator controls, paired at 10 and 20 images.
Independent of Experiments 01-04; reuses only model helpers in this benchmark.
"""
import argparse
from collections import Counter
import hashlib
import json
from itertools import product
import math
from pathlib import Path
import random

from build_sets import read_rules, compatible_rules
from pilot_10_vs_20 import load_model, evaluate, prompt_for, mapping

RULES = ("R1", "R2", "R3")
SEED = 2026
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def read_data(folder):
    metadata = Path(folder) / "metadata.json"
    records = json.loads(metadata.read_text())["records"]
    pool = [r for r in records if r["split"] == "train"]
    if len(records) != 1000 or len(pool) != 800:
        raise ValueError("Original main_72 dataset expects 1000 images, 800 in split=train")
    if any(not (Path(folder) / "images" / r["filename"]).is_file() for r in pool):
        raise ValueError("Some source images are missing")
    return pool, hashlib.sha256(metadata.read_bytes()).hexdigest()


def satisfies(rule, bits):
    a, b, c = map(bool, bits)
    return {"R1": (a and b) or c, "R2": (a and c) or b,
            "R3": (b and c) or a}[rule]


def original_ten(pool, rule, index, seed):
    """Match original 01-operator-controls.sample_set(coverage, n=10) exactly."""
    rng = random.Random(seed + 1000 * RULES.index(rule) + index)
    admissible = [b for b in product((0, 1), repeat=3) if satisfies(rule, b)]
    eligible = [r for r in pool if satisfies(rule, r["bits"])]
    chosen = [rng.choice([r for r in eligible if tuple(r["bits"]) == b])
              for b in admissible]
    already = {r["filename"] for r in chosen}
    chosen += rng.sample([r for r in eligible if r["filename"] not in already], 5)
    rng.shuffle(chosen)
    return chosen


def paired_sets(pool, rules, count, seed):
    """Preserve historical ten-image order, then append ten new satisfying images."""
    if [r["id"] for r in rules] != list(RULES):
        raise ValueError("This original-compatible experiment requires R1, R2, R3")
    result = {}
    for rule in RULES:
        for index in range(count):
            ten = original_ten(pool, rule, index, seed)
            used = {r["filename"] for r in ten}
            eligible = [r for r in pool if satisfies(rule, r["bits"])
                        and r["filename"] not in used]
            rng = random.Random(seed + 800000 + 1000 * RULES.index(rule) + index)
            twenty = ten + rng.sample(eligible, 10)
            if len(set(r["filename"] for r in twenty)) != 20:
                raise ValueError("Duplicate image within extended set")
            if any(compatible_rules(
                ["".join(str(v) for v in r["bits"]) for r in seq], rules) != [rule]
                   for seq in (ten, twenty)):
                raise ValueError("Rule not uniquely identifiable")
            key = f"coverage:{rule}:{index}"
            result[key] = {
                "group_id": key, "true_rule": rule, "index": index,
                "10": [r["filename"] for r in ten],
                "20": [r["filename"] for r in twenty],
                "bits10": [r["bits"] for r in ten],
                "bits20": [r["bits"] for r in twenty],
            }
    return result


def entropy(values):
    counts = Counter(values)
    n = len(values)
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c)


def mi(truth, predicted):
    n = len(truth)
    a, b = Counter(truth), Counter(predicted)
    joint = Counter(zip(truth, predicted))
    return sum((v/n)*math.log2(v*n/(a[x]*b[y]))
               for (x, y), v in joint.items())


def summarize(rows):
    result = {}
    for size in (10, 20):
        sub = [r for r in rows if r["images_per_set"] == size]
        if not sub:
            continue
        rng = random.Random(2026)
        groups = {}
        for r in sub:
            groups.setdefault(r["group_id"], []).append(r)
        truth = [r["true_rule"] for r in sub]
        pred = [r["predicted_rule"] for r in sub]
        observed_mi = mi(truth, pred)
        null = []
        group_ids = list(groups)
        group_rules = [groups[k][0]["true_rule"] for k in group_ids]
        for _ in range(1000):
            shuffled = group_rules.copy()
            rng.shuffle(shuffled)
            perm = dict(zip(group_ids, shuffled))
            null.append(mi([perm[r["group_id"]] for r in sub], pred))
        boot = []
        for _ in range(1000):
            chosen = rng.choices(group_ids, k=len(group_ids))
            sample = [r for k in chosen for r in groups[k]]
            boot.append(sum(r["correct"] for r in sample) / len(sample))
        boot.sort()
        correct = sum(r["correct"] for r in sub)
        result[str(size)] = {
            "trials": len(sub),
            "sets": len(groups),
            "accuracy": correct/len(sub),
            "chance": 1/3,
            "accuracy_95ci_cluster": [boot[24], boot[975]],
            "per_rule": {
                rule: sum(r["correct"] for r in sub if r["true_rule"] == rule)
                      / sum(r["true_rule"] == rule for r in sub)
                for rule in RULES
            },
            "answer_labels": dict(Counter(r["predicted_label"] for r in sub)),
            "predicted_rules": dict(Counter(pred)),
            "confusion": {
                rule: dict(Counter(r["predicted_rule"] for r in sub
                                   if r["true_rule"] == rule)) for rule in RULES
            },
            "mutual_information_bits": observed_mi,
            "mi_null_mean": sum(null)/len(null),
            "mi_permutation_p": (1+sum(v >= observed_mi for v in null))/(len(null)+1),
            "mean_true_probability": sum(r["probabilities"][r["true_label"]]
                                         for r in sub)/len(sub),
            "mean_nll": sum(-math.log(max(r["probabilities"][r["true_label"]], 1e-12))
                            for r in sub)/len(sub),
            "output_entropy_bits": entropy(pred),
            "rotation_consistency": sum(len({r["predicted_rule"] for r in group}) == 1
                                        for group in groups.values())/len(groups),
        }
    if "10" in result and "20" in result:
        result["paired_accuracy_difference_20_minus_10"] = (
            result["20"]["accuracy"] - result["10"]["accuracy"])
    result["caution"] = ("Answer rotations reuse image sets; groups, not trials, "
                         "are the unit for bootstrap uncertainty. The image pool "
                         "is the original experiment's 'train' partition, but "
                         "no model training occurs in operator controls.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rules", type=Path, default=Path(__file__).with_name("rules.json"))
    p.add_argument("--sets-per-rule", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-pixels", type=int, default=128*28*28)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if args.sets_per_rule < 1:
        p.error("--sets-per-rule must be positive")
    rules = read_rules(args.rules)["rules"]
    pool, data_sha = read_data(args.data_dir)
    sets = paired_sets(pool, rules, args.sets_per_rule, args.seed)
    print(f"Original-compatible sets: {len(sets)} pairs; "
          f"{len(sets)*3} trials per image count", flush=True)
    if args.validate_only:
        print("Validated without loading Qwen.")
        return

    config = {
        "protocol": "original-main_72-coverage-cyclic-v1",
        "model": MODEL, "revision": REVISION, "quantization": "4bit-NF4",
        "data_sha256": data_sha,
        "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest(),
        "seed": args.seed, "sets_per_rule": args.sets_per_rule,
        "image_counts": [10, 20], "rotations": 3,
        "max_pixels": args.max_pixels, "prompt": "identical to original main_72",
        "readout": "next token restricted X/Y/Z softmax; thinking=False",
        "sampling": ("original coverage set for 10 images; same original "
                     "10-image prefix + 10 new positive images for 20"),
    }
    if not args.smoke_test:
        args.out.mkdir(parents=True, exist_ok=True)
        cfg = args.out / "config.json"
        if cfg.exists() and json.loads(cfg.read_text()) != config:
            raise ValueError("Existing output has different settings")
        cfg.write_text(json.dumps(config, indent=2)+"\n")
        manifest = args.out / "paired_sets.json"
        if manifest.exists() and json.loads(manifest.read_text()) != sets:
            raise ValueError("Saved image-set manifest differs")
        manifest.write_text(json.dumps(sets, indent=2)+"\n")
        log = args.out / "runs.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()
                if line.strip()] if log.exists() else []
        seen = {r["trial_id"] for r in rows}
        if len(seen) != len(rows):
            raise ValueError("Duplicate trial IDs in saved output")
    else:
        log, rows, seen = None, [], set()
    jobs = []
    for item in sets.values():
        for rotation in range(3):
            for size in (10, 20):
                jobs.append((f"{item['group_id']}:cyclic:{rotation}:images:{size}",
                             item, rotation, size))
    if not args.smoke_test and not seen <= {job[0] for job in jobs}:
        raise ValueError("Unexpected completed trials")
    if args.smoke_test:
        jobs = [next(job for job in jobs if job[3] == 20)]
    incomplete = [job for job in jobs if job[0] not in seen]
    if incomplete:
        model, processor, token_ids = load_model()
        import torch
        for number, (key, item, rotation, size) in enumerate(incomplete, 1):
            labels = mapping(list(RULES), rotation)
            prompt = prompt_for(rules, ["A", "B", "C"], labels, size)
            torch.cuda.reset_peak_memory_stats()
            probs, tokens = evaluate(model, processor, token_ids, item[str(size)],
                                      args.data_dir, prompt, args.max_pixels)
            chosen = max("XYZ", key=lambda label: probs[label])
            truth_label = next(label for label in "XYZ"
                               if labels[label] == item["true_rule"])
            row = {
                "trial_id": key, "group_id": item["group_id"],
                "hidden_rule": item["true_rule"], "true_rule": item["true_rule"],
                "set_index": item["index"], "images_per_set": size,
                "rotation": rotation, "image_files": item[str(size)],
                "scene_bits": item["bits"+str(size)],
                "sampling": "coverage", "identifiable": True,
                "label_mapping": labels, "display_order": list("XYZ"),
                "modality": "images", "prompt": prompt,
                "probabilities": probs, "chosen_label": chosen,
                "chosen_rule": labels[chosen], "predicted_label": chosen,
                "predicted_rule": labels[chosen], "true_label": truth_label,
                "chosen_position": "XYZ".index(chosen)+1,
                "correct": labels[chosen] == item["true_rule"],
                "true_probability": probs[truth_label],
                "input_tokens": tokens,
                "peak_gpu_gib": round(torch.cuda.max_memory_allocated()/2**30, 3),
            }
            print(f"{len(seen)+number}/{len(jobs)} {key} "
                  f"truth={item['true_rule']} predicted={labels[chosen]} "
                  f"{'OK' if row['correct'] else 'wrong'}", flush=True)
            if args.smoke_test:
                print("20-image smoke test passed; no results saved.")
                return
            with log.open("a") as file:
                file.write(json.dumps(row)+"\n")
            rows.append(row)
    if not args.smoke_test:
        if len(rows) != len(jobs):
            raise ValueError("Missing results; rerun to resume")
        summary = summarize(rows)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
