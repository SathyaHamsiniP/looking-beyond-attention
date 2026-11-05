"""Make fixed ten-image examples from the existing dataset. No images are created."""

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import random

RULES = ("R1", "R2", "R3")
BITS = list(product((0, 1), repeat=3))
COUNTS = {"train": 30, "validation": 10, "test": 10}


def satisfies(rule, bits):
    a, b, c = map(bool, bits)
    return {"R1": (a and b) or c, "R2": (a and c) or b, "R3": (b and c) or a}[rule]


def fingerprint(data):
    h = hashlib.sha256((data / "metadata.json").read_bytes())
    for path in sorted((data / "images").glob("*.png")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build(data, seed=2026):
    data = Path(data).resolve()
    records = json.loads((data / "metadata.json").read_text())["records"]
    if len(records) != 1000 or len({r["filename"] for r in records}) != 1000:
        raise ValueError("Expected 1,000 unique image records")
    if {p.name for p in (data / "images").glob("*.png")} != {
        r["filename"] for r in records
    }:
        raise ValueError("Images differ from metadata")
    rng = random.Random(seed)
    partitions = {}
    for index, bits in enumerate(BITS):
        train = sorted(
            r["filename"]
            for r in records
            if tuple(r["bits"]) == bits and r["split"] == "train"
        )
        held = sorted(
            r["filename"]
            for r in records
            if tuple(r["bits"]) == bits and r["split"] == "test"
        )
        if len(train) != 100 or len(held) != 25:
            raise ValueError(
                "Expected 100 training and 25 held-out images per configuration"
            )
        rng.shuffle(held)
        n = 12 + index % 2
        partitions.update({name: "train" for name in train})
        partitions.update(
            {name: "validation" if i < n else "test" for i, name in enumerate(held)}
        )
    sets = []
    for partition, count in COUNTS.items():
        seen = set()
        for rule in RULES:
            positive = [b for b in BITS if satisfies(rule, b)]
            pool = [
                r
                for r in records
                if partitions[r["filename"]] == partition and satisfies(rule, r["bits"])
            ]
            for index in range(count):
                for attempt in range(10000):
                    chosen = [
                        rng.choice([r for r in pool if tuple(r["bits"]) == bits])
                        for bits in positive
                    ]
                    used = {r["filename"] for r in chosen}
                    chosen += rng.sample(
                        [r for r in pool if r["filename"] not in used], 5
                    )
                    identity = tuple(sorted(r["filename"] for r in chosen))
                    if identity not in seen:
                        break
                else:
                    raise ValueError(
                        "Could not build a new set without relaxing coverage"
                    )
                seen.add(identity)
                rng.shuffle(chosen)
                # Balance the correct answer letter across the 30 training examples per rule.
                rotation = index % 3
                order = RULES[rotation:] + RULES[:rotation]
                sets.append(
                    dict(
                        set_id=f"{partition}_{rule}_{index:02d}",
                        partition=partition,
                        rule=rule,
                        image_files=[r["filename"] for r in chosen],
                        bits=[r["bits"] for r in chosen],
                        option_mapping=dict(zip("XYZ", order)),
                        target_label="XYZ"[order.index(rule)],
                    )
                )
    reuse = Counter((s["partition"], f) for s in sets for f in s["image_files"])
    for s in sets:
        s["reused_image_files"] = [
            f for f in s["image_files"] if reuse[s["partition"], f] > 1
        ]
    return dict(
        seed=seed,
        dataset_fingerprint=fingerprint(data),
        image_partitions=partitions,
        sets=sets,
        validation=dict(
            image_counts=dict(Counter(partitions.values())),
            set_counts=dict(Counter(s["partition"] for s in sets)),
            coverage="all five satisfying configurations in each set",
            unique_set_identity="sorted image filenames; order does not make a new set",
        ),
    )


def load(path, data):
    value = json.loads(Path(path).read_text())
    # Rebuild from source rather than trusting saved labels or partition claims.
    if value != build(Path(data), value["seed"]):
        raise ValueError("Manifest differs from the fixed construction or dataset")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    value = (
        load(a.manifest, a.data_dir) if a.validate_only else build(a.data_dir, a.seed)
    )
    if not a.validate_only:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        if a.manifest.exists() and json.loads(a.manifest.read_text()) != value:
            raise ValueError("Choose a new manifest path; existing file differs")
        a.manifest.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value["validation"], indent=2))


if __name__ == "__main__":
    main()
