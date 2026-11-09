"""Frozen Qwen: original R1-R3 test sets with three versus seven candidates.

CPU validation never loads a model or writes output. Smoke runs one paired set
at rotation zero and saves nothing. --pilot saves one rotation of all 30 sets;
default evaluation balances every label using 3/7 rotations. --resume explicitly
reopens only this runner's compatible output. No datasets/manifests are generated.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path

from build_sets import compatible_rules, read_json, read_rules, sha256_file
import pilot_10_vs_20 as shared

IDS = [f"R{i}" for i in range(1, 8)]
EXPRESSIONS = ["(A AND B) OR C", "(A AND C) OR B", "(B AND C) OR A",
               "(A XOR B) OR C", "(A XOR C) OR B", "(B XOR C) OR A",
               "(A OR B OR C) AND NOT(A AND B AND C)"]
CAUTION = ("Tests the original three generating rules under different candidate lists, "
           "NOT accuracy across all seven generating rules. Test sets were previously "
           "used in exploratory analysis and are not an untouched final benchmark. "
           "Rotations are repeated measurements; aggregate and pair by image set. "
           "Images may also be reused across sets.")


def truth_tables():
    tables = [[] for _ in IDS]
    for a, b, c in product((False, True), repeat=3):
        values = [(a and b) or c, (a and c) or b, (b and c) or a,
                  (a != b) or c, (a != c) or b, (b != c) or a,
                  (a or b or c) and not (a and b and c)]
        bits = ''.join(str(int(x)) for x in (a, b, c))
        for table, value in zip(tables, values):
            if value:
                table.append(bits)
    return {"operand_names": ["A", "B", "C"], "rules": [
        dict(id=rid, expression=expr, true_bits=table)
        for rid, expr, table in zip(IDS, EXPRESSIONS, tables)]}


def validate_rules(path):
    spec = read_rules(path)  # rejects duplicate IDs/tables and invalid bits
    if spec != truth_tables():
        raise ValueError("Seven-rule definitions/truth tables differ from the specified Boolean formulas")
    original = read_rules(Path(__file__).with_name("rules.json"))
    if original != dict(operand_names=spec["operand_names"], rules=spec["rules"][:3]):
        raise ValueError("R1-R3 must match existing rules.json exactly")
    return spec


def validate(data_dir, manifest_path, rules_path, partitions_path=None):
    spec = validate_rules(rules_path)
    m = read_json(manifest_path)
    original_rules = Path(__file__).with_name("rules.json")
    if m["images_per_set"] != 10 or m["rule_ids"] != IDS[:3] or m["operand_names"] != ["A", "B", "C"]:
        raise ValueError("Expected the existing three-rule ten-image probe manifest")
    metadata = Path(data_dir) / "metadata.json"
    if sha256_file(metadata) != m["source_metadata_sha256"]:
        raise ValueError("Dataset metadata SHA-256 differs from the saved manifest")
    if sha256_file(original_rules) != m["rules_sha256"]:
        raise ValueError("Manifest rules hash differs from existing rules.json")
    records = read_json(metadata)["records"]
    images = {r["filename"]: r for r in records}
    if len(records) != 1000 or len(images) != 1000:
        raise ValueError("Expected 1,000 unique dataset image records")
    if m["partitions_sha256"] is not None:
        if partitions_path is None or sha256_file(partitions_path) != m["partitions_sha256"]:
            raise ValueError("Manifest uses external partitions: supply its exact --partitions file")
        assignments = read_json(partitions_path)
        assignments = assignments.get("image_partitions", assignments)
        if set(assignments) != set(images):
            raise ValueError("Partition file must assign every source image")
    else:
        if partitions_path is not None:
            raise ValueError("Do not override the manifest's metadata partitions")
        assignments = {name: r["split"] for name, r in images.items()}
    if dict(Counter(assignments.values())) != m["image_partition_counts"]:
        raise ValueError("Image partition counts differ from manifest")
    for name, row in images.items():
        if Path(name).name != name or not (Path(data_dir) / "images" / name).is_file():
            raise ValueError(f"Missing or invalid image filename: {name}")
        if len(row["bits"]) != 3 or any(type(x) is not int or x not in (0, 1) for x in row["bits"]):
            raise ValueError(f"Invalid A/B/C configuration in metadata: {name}")
    ids = [s["set_id"] for s in m["sets"]]
    if len(set(ids)) != len(ids) or len(ids) != m["set_count"]:
        raise ValueError("Duplicate set IDs or incorrect manifest set_count")
    selected = [s for s in m["sets"] if s["split"] == "test"]
    if len(selected) != 30 or Counter(s["rule"] for s in selected) != {r: 10 for r in IDS[:3]}:
        raise ValueError("Expected exactly 30 test sets, ten each for R1-R3")
    identities = set()
    for item in selected:
        sid, names, bits = item["set_id"], item["image_files"], item["bits"]
        if len(names) != 10 or len(set(names)) != 10 or len(bits) != 10:
            raise ValueError(f"{sid}: expected ten distinct ordered images and configurations")
        identity = tuple(sorted(names))
        if identity in identities:
            raise ValueError(f"{sid}: duplicate image set")
        identities.add(identity)
        for name, bit in zip(names, bits):
            if name not in images or assignments[name] != "test":
                raise ValueError(f"{sid}: {name} missing or outside the existing test partition")
            if bit != ''.join(map(str, images[name]["bits"])):
                raise ValueError(f"{sid}: saved configuration differs from metadata for {name}")
        for n in (3, 7):
            matches = compatible_rules(bits, spec["rules"][:n])
            if matches != [item["rule"]]:
                raise ValueError(f"{sid}: not uniquely identifiable among {n} rules; "
                                 f"intended={item['rule']}, compatible={matches}, observed={bits}")
    return selected, spec


def mapping(n, rotation):
    if n not in (3, 7) or not 0 <= rotation < n:
        raise ValueError("Invalid candidate count or rotation")
    order = IDS[:n]
    return dict(zip(map(str, range(1, n + 1)), order[rotation:] + order[:rotation]))


def prompt_for(spec, current):
    options = '\n'.join(f"{label}: {next(r['expression'] for r in spec['rules'] if r['id'] == rid)}"
                        for label, rid in current.items())
    # Only options and permitted answer labels differ between conditions.
    return ("Images 1-10 all satisfy ONE hidden Boolean rule.\n\n"
            "A is true when a green cube is present.\n"
            "B is true when a blue sphere is present.\n"
            "C is true when a red cylinder is present.\n"
            "Each concept requires the exact colour and shape.\n"
            "Other objects are irrelevant distractors.\n\n"
            f"Candidate rules:\n{options}\n\n"
            "AND requires both inputs to be true.\n"
            "OR requires at least one input to be true, including both.\n"
            "XOR requires exactly one input to be true.\n"
            "NOT reverses the truth value of its input.\n"
            "Follow the parentheses.\n\n"
            "Pick the rule consistent with every image.\n"
            f"Reply with nothing but its numeric label: {', '.join(current)}.\nANSWER:")


def label_tokens(tokenizer):
    encoded = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 8)]
    if any(len(x) != 1 for x in encoded) or len({x[0] for x in encoded}) != 7:
        raise ValueError("Numeric labels 1-7 must be distinct single tokens under the actual tokenizer")
    return [x[0] for x in encoded]


def evaluate_numeric(model, processor, token_ids, files, data_dir, prompt, max_pixels):
    """Numeric readout with the existing helper's exact preprocessing and forward.

    That helper hardcodes XYZ when returning scores. Keep it unchanged and
    extend the readout here; model loading is shared directly.
    """
    import torch
    from PIL import Image
    content = []
    for name in files:
        with Image.open(Path(data_dir) / "images" / name) as file:
            picture = file.convert("RGB")
        if picture.width * picture.height > max_pixels:
            scale = (max_pixels / (picture.width * picture.height)) ** .5
            picture = picture.resize((max(1, int(picture.width * scale)),
                                      max(1, int(picture.height * scale))), Image.Resampling.LANCZOS)
        content.append({"type": "image", "image": picture})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    # Check labels in the actual rendered assistant-prefix context as well.
    rendered = processor.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    prefix = processor.tokenizer.encode(rendered, add_special_tokens=False)
    for label, tid in zip(map(str, range(1, len(token_ids) + 1)), token_ids):
        if processor.tokenizer.encode(rendered + label, add_special_tokens=False) != prefix + [tid]:
            raise ValueError(f"Label {label} is not one append-only token at the answer boundary")
    prepared = processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**prepared, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        scores = logits[token_ids]
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite numeric-label logits")
        probs = scores.softmax(dim=0).cpu().tolist()
    return dict(zip(map(str, range(1, len(token_ids) + 1)), probs)), int(prepared["input_ids"].shape[1])


def jobs(sets, spec, single=False):
    result = {}
    for item in sets:
        for n in (3, 7):
            for rotation in range(1 if single else n):
                current = mapping(n, rotation)
                key = f"{item['set_id']}:{n}:{rotation}"
                result[key] = dict(key=key, set_id=item["set_id"], condition=n,
                    rotation=rotation, true_rule=item["rule"], image_files=item["image_files"].copy(),
                    bits=item["bits"].copy(), option_mapping=current, prompt=prompt_for(spec, current))
    return result


def validate_row(row, job):
    for field, value in job.items():
        if row.get(field) != value:
            raise ValueError(f"Saved {job['key']}: mismatched {field}")
    probs = row["probabilities"]
    if set(probs) != set(job["option_mapping"]) or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()) or not math.isclose(sum(probs.values()), 1, abs_tol=1e-6):
        raise ValueError(f"Saved {job['key']}: invalid probabilities")
    label = max(job["option_mapping"], key=probs.get)
    predicted = job["option_mapping"][label]
    if row["predicted_label"] != label or row["predicted_rule"] != predicted or type(row["correct"]) is not bool or row["correct"] != (predicted == job["true_rule"]):
        raise ValueError(f"Saved {job['key']}: prediction/mapping/correctness disagree")


def summarize(rows, single=False):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["set_id"])].append(row)
    result = {"caution": CAUTION, "unit": "image set; accuracy averaged over rotations within each set",
              "conditions": {}, "paired_sets": []}
    scores = {}
    for n in (3, 7):
        sub = {sid: group for (condition, sid), group in groups.items() if condition == n}
        expected = 1 if single else n
        if not sub or any(len(g) != expected for g in sub.values()):
            raise ValueError("Summary requires complete rotations for each included set")
        scores[n] = {sid: sum(r["correct"] for r in g) / len(g) for sid, g in sub.items()}
        accuracy = sum(scores[n].values()) / len(sub)
        by_rule = {rule: [sid for sid, g in sub.items() if g[0]["true_rule"] == rule] for rule in IDS[:3]}
        counts = Counter(r["predicted_label"] for g in sub.values() for r in g)
        # Fractional confusion entries give each set total weight one.
        confusion = {rule: {pred: sum(sum(r["predicted_rule"] == pred for r in sub[sid]) / len(sub[sid])
                                      for sid in by_rule[rule]) for pred in IDS[:n]} for rule in IDS[:3]}
        consistent = sum(len({r["predicted_rule"] for r in g}) == 1 for g in sub.values())
        result["conditions"][str(n)] = dict(sets=len(sub), evaluations=sum(map(len, sub.values())),
            accuracy=accuracy, chance=1/n, chance_adjusted_accuracy=(accuracy-1/n)/(1-1/n),
            per_rule_accuracy={rule: sum(scores[n][sid] for sid in ids)/len(ids) if ids else None for rule, ids in by_rule.items()},
            confusion_set_equivalents=confusion,
            confusion_row_fractions={rule: {p: v/len(by_rule[rule]) if by_rule[rule] else None for p, v in values.items()} for rule, values in confusion.items()},
            answer_position_counts={str(i): counts[str(i)] for i in range(1, n+1)},
            answer_position_fractions={str(i): counts[str(i)]/sum(counts.values()) for i in range(1,n+1)},
            rule_consistent_sets=None if single else consistent,
            rotation_consistency=None if single else consistent/len(sub),
            rotation_consistency_note="Not measured with a single rotation" if single else "Same predicted rule across every rotation; correctness not required")
    if set(scores[3]) != set(scores[7]):
        raise ValueError("Conditions must contain the same paired image sets")
    for sid in scores[3]:
        left, right = groups[(3, sid)], groups[(7, sid)]
        if left[0]["image_files"] != right[0]["image_files"]:
            raise ValueError(f"Paired images/order differ: {sid}")
        result["paired_sets"].append(dict(set_id=sid, true_rule=left[0]["true_rule"],
            accuracy_3=scores[3][sid], accuracy_7=scores[7][sid], difference_7_minus_3=scores[7][sid]-scores[3][sid]))
    deltas = [p["difference_7_minus_3"] for p in result["paired_sets"]]
    result["paired_comparison"] = dict(sets=len(deltas), mean_accuracy_difference_7_minus_3=sum(deltas)/len(deltas),
        improved_sets=sum(x > 1e-12 for x in deltas), worsened_sets=sum(x < -1e-12 for x in deltas),
        tied_sets=sum(abs(x) <= 1e-12 for x in deltas),
        chance_adjusted_difference_7_minus_3=result["conditions"]["7"]["chance_adjusted_accuracy"]-result["conditions"]["3"]["chance_adjusted_accuracy"])
    return result


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def resume_rows(out, config, required, resume):
    """Refuse existing folders by default; never truncate an interrupted log."""
    if not out.exists():
        if resume:
            raise ValueError("--resume needs an existing comparison results directory")
        return []
    if not resume:
        raise ValueError("Output already exists; choose a NEW directory or explicitly --resume this comparison")
    saved = read_json(out / "config.json")
    ids = saved.get("numeric_label_token_ids", [])
    if len(ids) != 7 or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != 7:
        raise ValueError("Saved numeric label token IDs are invalid")
    if {k: v for k, v in saved.items() if k != "numeric_label_token_ids"} != config:
        raise ValueError("Resume configuration differs (inputs, mode, source code or settings)")
    path = out / "predictions.jsonl"
    rows = []
    if path.exists():
        content = path.read_text()
        if content and not content.endswith("\n"):
            raise ValueError("Prediction log has a partial final line; preserve the file and inspect before resuming")
        seen = set()
        for line in content.splitlines():
            row = json.loads(line)
            key = row["key"]
            if key in seen or key not in required:
                raise ValueError(f"Duplicate or unexpected saved key: {key}")
            validate_row(row, required[key])
            rows.append(row); seen.add(key)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rules", type=Path, default=Path(__file__).with_name("rules_7.json"))
    p.add_argument("--partitions", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-pixels", type=int, default=128*28*28)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true", help="One rotation for all 30 paired sets")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.max_pixels < 1024:
        p.error("--max-pixels must be at least 1024")
    try:
        sets, spec = validate(args.data_dir, args.manifest, args.rules, args.partitions)
        print(f"Validated {len(sets)} existing test sets, ten per original rule, uniquely identifiable with 3 and 7 candidates.")
        print(CAUTION)
        if args.validate_only:
            return
        if args.resume and args.smoke_test:
            raise ValueError("Smoke test saves nothing and cannot resume")
        required = jobs(sets[:1] if args.smoke_test else sets, spec, args.pilot or args.smoke_test)
        config = dict(protocol="original-3-under-3-vs-7-v1", model=shared.MODEL, revision=shared.REVISION,
            quantization="4-bit NF4; double quantization; FP16 compute", thinking=False,
            manifest_sha256=sha256_file(args.manifest), metadata_sha256=sha256_file(args.data_dir/"metadata.json"),
            rules=spec, rules_sha256=sha256_file(args.rules), max_pixels=args.max_pixels,
            mode="pilot" if args.pilot else "balanced", scoring="restricted next-token numeric-label softmax",
            rotations={"3": 1 if args.pilot else 3, "7": 1 if args.pilot else 7},
            prompts={str(n): [prompt_for(spec, mapping(n, i)) for i in range(n)] for n in (3,7)},
            source_sha256={name: sha256_file(Path(__file__).with_name(name)) for name in ("compare_3_vs_7.py", "pilot_10_vs_20.py", "build_sets.py")},
            image_sha256={name: sha256_file(args.data_dir/"images"/name) for name in sorted({f for s in sets for f in s["image_files"]})},
            caution=CAUTION)
        if args.out.resolve() == args.data_dir.resolve() or args.data_dir.resolve() in args.out.resolve().parents or args.out.resolve() == args.manifest.parent.resolve():
            raise ValueError("Use a separate NEW results directory, not the dataset or manifest directory")
        rows = [] if args.smoke_test else resume_rows(args.out, config, required, args.resume)
        done = {r["key"] for r in rows}
        pending = [job for key, job in required.items() if key not in done]
        print(f"{len(required)} total model evaluations; {len(pending)} remaining.", flush=True)
        if pending:
            model, processor, _ = shared.load_model()
            token_ids = label_tokens(processor.tokenizer)
            if not args.smoke_test:
                if args.out.exists():
                    if read_json(args.out/"config.json")["numeric_label_token_ids"] != token_ids:
                        raise ValueError("Actual tokenizer label IDs differ from saved configuration")
                else:
                    args.out.mkdir(parents=True, exist_ok=False)
                    atomic_json(args.out/"config.json", dict(config, numeric_label_token_ids=token_ids))
            for job in pending:
                probs, tokens = evaluate_numeric(model, processor, token_ids[:job["condition"]],
                    job["image_files"], args.data_dir, job["prompt"], args.max_pixels)
                label = max(job["option_mapping"], key=probs.get)
                rule = job["option_mapping"][label]
                row = dict(job, probabilities=probs, predicted_label=label, predicted_rule=rule,
                           correct=rule==job["true_rule"], input_tokens=tokens)
                validate_row(row, job)
                if not args.smoke_test:
                    with (args.out/"predictions.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                        stream.flush(); os.fsync(stream.fileno())
                rows.append(row)
                print(f"{row['key']}: true={row['true_rule']} predicted={rule} label={label}", flush=True)
        report = summarize(rows, args.pilot or args.smoke_test)
        if args.smoke_test:
            print("Paired smoke test complete; no results saved.")
        else:
            atomic_json(args.out/"summary.json", report)
            print(f"Results: {args.out}")
        print(json.dumps(report, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
