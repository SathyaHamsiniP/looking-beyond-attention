"""Build derived image splits and balanced rule sets without altering dataset-v1."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

try:
    from .rules import BITS, EXPRESSIONS, consistent, evaluate, library, truth_tables
except ImportError:
    from rules import BITS, EXPRESSIONS, consistent, evaluate, library, truth_tables

VERSION_NAMES = ("v1-positive-3rules", "v2-mixed-3rules", "v3-mixed-varied-rules")
CHECKSUM = "86b91e44d617cb7d9ffc893c4459e5e5eb621841eb98f36f166d7e67dfd2f030"


def split_records(records, seed=2026):
    if len(records) != 1000 or len({r["filename"] for r in records}) != 1000:
        raise ValueError("Expected 1,000 unique dataset-v1 records")
    if Counter(r["split"] for r in records) != {"train": 800, "test": 200}:
        raise ValueError("Original split must be 800 train / 200 held out")
    rng = random.Random(seed)
    split = {r["filename"]: "train" for r in records if r["split"] == "train"}
    for i, bits in enumerate(BITS):
        train = [
            r for r in records if tuple(r["bits"]) == bits and r["split"] == "train"
        ]
        held = sorted(
            r["filename"]
            for r in records
            if tuple(r["bits"]) == bits and r["split"] == "test"
        )
        if len(train) != 100 or len(held) != 25:
            raise ValueError(
                "Expected 100 train and 25 held-out images per configuration"
            )
        rng.shuffle(held)
        n = 12 + i % 2
        split.update(
            {name: "validation" if j < n else "test" for j, name in enumerate(held)}
        )
    return split


def validate_item(item, rules, version):
    obs = item["observations"]
    if len({o["filename"] for o in obs}) != len(obs):
        raise ValueError("Duplicate image in set")
    if any(
        type(o["label"]) is not bool or evaluate(item["rule"], o["bits"]) != o["label"]
        for o in obs
    ):
        raise ValueError("Incorrect labels")
    if consistent(obs, rules) != [item["rule"]]:
        raise ValueError("Ambiguous or incorrect generating rule")
    singletons = [i for i, o in enumerate(obs) if len(consistent([o], rules)) == 1]
    if not version.startswith("v1"):
        if singletons:
            raise ValueError("Singleton identifies rule")
        if sum(o["label"] for o in obs) * 2 != len(obs):
            raise ValueError("Mixed tasks require equal label counts")
    elif not all(o["label"] for o in obs):
        raise ValueError("V1 must be positive-only")
    if sorted(item["option_order"]) != sorted(rules):
        raise ValueError("Invalid option order")
    return dict(
        unique_rule=True,
        singleton_indices=singletons,
        positive_count=sum(o["label"] for o in obs),
        configuration_counts=dict(Counter("".join(map(str, o["bits"])) for o in obs)),
    )


def build(data_dir, version, seed=2026, sets_per_rule=12, examples=10):
    if version not in VERSION_NAMES:
        raise ValueError("Unknown task version")
    data_dir = Path(data_dir).resolve()
    metadata = (data_dir / "metadata.json").read_bytes()
    records = json.loads(metadata)["records"]
    split = split_records(records, seed)
    if examples < 4 or examples % 2 or examples > 20 or sets_per_rule < 1:
        raise ValueError("Use an even set size 4..20 and positive sets-per-rule")
    rules = library(version)
    rng = random.Random(seed)
    sets = []
    for partition in ("train", "validation", "test"):
        for rule in rules:
            eligible = [
                r
                for r in records
                if split[r["filename"]] == partition
                and (
                    evaluate(rule, r["bits"])
                    if version.startswith("v1")
                    else sum(
                        evaluate(other, r["bits"]) == evaluate(rule, r["bits"])
                        for other in rules
                    )
                    > 1
                )
            ]
            for index in range(sets_per_rule):
                for attempt in range(10000):
                    if version.startswith("v1"):
                        if examples < 5:
                            raise ValueError("V1 coverage needs at least five examples")
                        chosen = [
                            rng.choice([r for r in eligible if tuple(r["bits"]) == b])
                            for b in BITS
                            if evaluate(rule, b)
                        ]
                        used = {r["filename"] for r in chosen}
                        chosen += rng.sample(
                            [r for r in eligible if r["filename"] not in used],
                            examples - len(chosen),
                        )
                    else:
                        chosen = []
                        for label in (False, True):
                            pool = [
                                r
                                for r in eligible
                                if evaluate(rule, r["bits"]) == label
                            ]
                            if len(pool) < examples // 2:
                                raise ValueError(
                                    f"Insufficient singleton-safe observations: {rule}"
                                )
                            chosen += rng.sample(pool, examples // 2)
                    rng.shuffle(chosen)
                    order = rng.sample(rules, len(rules))
                    item = dict(
                        set_id=f"{partition}-{rule}-{index:05d}",
                        partition=partition,
                        rule=rule,
                        seed=seed,
                        option_order=order,
                        examples_per_set=examples,
                        observations=[
                            dict(
                                filename=r["filename"],
                                path="images/" + r["filename"],
                                bits=r["bits"],
                                label=evaluate(rule, r["bits"]),
                            )
                            for r in chosen
                        ],
                    )
                    try:
                        item["validation"] = validate_item(item, rules, version)
                    except ValueError:
                        continue
                    sets.append(item)
                    break
                else:
                    raise ValueError(f"Cannot construct an informative set for {rule}")
    counts = Counter(
        (s["partition"], o["filename"]) for s in sets for o in s["observations"]
    )
    for item in sets:
        item["images_reused_across_sets"] = any(
            counts[item["partition"], o["filename"]] > 1 for o in item["observations"]
        )
    return dict(
        schema_version=1,
        task_version=version,
        dataset_version="dataset-v1",
        dataset_sha256=CHECKSUM,
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        seed=seed,
        rules={r: EXPRESSIONS[r] for r in rules},
        truth_tables=truth_tables(rules),
        split_manifest=split,
        sets=sets,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--task", choices=VERSION_NAMES, default=VERSION_NAMES[2])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sets-per-rule", type=int, default=12)
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    value = build(a.data_dir, a.task, a.seed, a.sets_per_rule, a.examples)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists() and json.loads(a.output.read_text()) != value:
        raise ValueError("Refusing to overwrite a different manifest")
    a.output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Saved {len(value['sets'])} sets to {a.output}")


if __name__ == "__main__":
    main()
