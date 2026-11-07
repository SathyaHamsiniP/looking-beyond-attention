"""Build image-set manifests. Self-contained; Python standard library only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rules(path):
    spec = read_json(path)
    names = spec["operand_names"]
    if not 2 <= len(names) <= 8 or len(names) != len(set(names)):
        raise ValueError("Expected 2-8 distinct operand names")
    universe = {format(i, f"0{len(names)}b") for i in range(2 ** len(names))}
    rules = spec["rules"]
    ids = [r["id"] for r in rules]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Rule IDs must be distinct")
    for r in rules:
        bits = r["true_bits"]
        if not bits or len(bits) != len(set(bits)) or not set(bits) <= universe:
            raise ValueError(f"Invalid truth table for {r['id']}")
        if not r.get("expression"):
            raise ValueError(f"Missing readable expression for {r['id']}")
    if len({tuple(sorted(r["true_bits"])) for r in rules}) != len(rules):
        raise ValueError("Duplicate Boolean truth tables")
    return spec


def compatible_rules(observed, rules):
    """Positive-only: compatible if every observed configuration satisfies rule."""
    seen = set(observed)
    return [r["id"] for r in rules if seen <= set(r["true_bits"])]


def load_images(data_dir, partitions_path, width):
    folder = Path(data_dir)
    metadata_path = folder / "metadata.json"
    records = read_json(metadata_path)["records"]
    names = [r["filename"] for r in records]
    if not records or len(names) != len(set(names)):
        raise ValueError("Metadata must list unique image filenames")
    if partitions_path:
        assignments = read_json(partitions_path)
        assignments = assignments.get("image_partitions", assignments)
        if set(assignments) != set(names):
            raise ValueError("Partition file must assign every image exactly once")
    else:
        if not all("split" in r for r in records):
            raise ValueError("Supply --partitions or existing split in metadata")
        assignments = {r["filename"]: r["split"] for r in records}
    images = []
    for r in records:
        bits = r["bits"]
        if len(bits) != width or any(type(x) is not int or x not in (0, 1) for x in bits):
            raise ValueError(f"Invalid bits for {r['filename']}")
        if Path(r["filename"]).name != r["filename"]:
            raise ValueError("Image filenames must be flat")
        if not (folder / "images" / r["filename"]).is_file():
            raise ValueError(f"Missing image: {r['filename']}")
        split = assignments[r["filename"]]
        if not isinstance(split, str) or not split:
            raise ValueError("Each image needs a nonempty partition name")
        images.append({"name": r["filename"], "bits": "".join(map(str, bits)), "split": split})
    return images, sha256_file(metadata_path)


def make_rng(seed, split, rule, index):
    key = f"{seed}|{split}|{rule}|{index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def make_manifest(data_dir, rules_path, partitions_path, counts, images_per_set, seed):
    """Counts are number of sets PER RULE within each partition."""
    spec = read_rules(rules_path)
    rules = spec["rules"]
    if not counts or any(n < 1 for n in counts.values()):
        raise ValueError("Provide positive sets-per-rule counts")
    if images_per_set < 1:
        raise ValueError("Images per set must be positive")
    images, metadata_sha = load_images(data_dir, partitions_path, len(spec["operand_names"]))
    by_split = {}
    for row in images:
        by_split.setdefault(row["split"], []).append(row)
    if not set(counts) <= set(by_split):
        raise ValueError("Requested partition missing from images")
    result = []
    for split, count in counts.items():
        for rule in rules:
            positive = sorted(rule["true_bits"])
            if images_per_set < len(positive):
                raise ValueError(f"{rule['id']} needs {len(positive)} images for coverage")
            pools = {b: sorted((x for x in by_split[split] if x["bits"] == b),
                                key=lambda x: x["name"]) for b in positive}
            if any(not pool for pool in pools.values()):
                raise ValueError(f"Missing satisfying configuration for {rule['id']} in {split}")
            eligible = sorted((x for pool in pools.values() for x in pool),
                              key=lambda x: x["name"])
            if images_per_set > len(eligible):
                raise ValueError(f"Not enough distinct images for {rule['id']} in {split}")
            identities = set()
            for index in range(count):
                rng = make_rng(seed, split, rule["id"], index)
                chosen = [rng.choice(pools[b]) for b in positive]
                used = {x["name"] for x in chosen}
                remainder = [x for x in eligible if x["name"] not in used]
                rng.shuffle(remainder)
                chosen += remainder[:images_per_set - len(chosen)]
                filenames = [x["name"] for x in chosen]
                identity = tuple(sorted(filenames))
                if identity in identities:
                    raise ValueError(f"Duplicate image set for {split}/{rule['id']}")
                identities.add(identity)
                matched = compatible_rules([x["bits"] for x in chosen], rules)
                if matched != [rule["id"]]:
                    raise ValueError(
                        f"{rule['id']} is NOT uniquely identifiable among candidates "
                        f"in {split}; compatible rules: {matched}"
                    )
                result.append({"set_id": f"{split}_{rule['id']}_{index:04d}",
                               "split": split, "rule": rule["id"],
                               "image_files": filenames, "bits": [x["bits"] for x in chosen]})
    return {
        "format_version": 1,
        "status": "draft; benchmark protocol pending alignment",
        "seed": seed,
        "operand_names": spec["operand_names"],
        "rule_ids": [r["id"] for r in rules],
        "images_per_set": images_per_set,
        "sets_per_rule_by_partition": counts,
        "source_metadata_sha256": metadata_sha,
        "rules_sha256": sha256_file(rules_path),
        "partitions_sha256": sha256_file(partitions_path) if partitions_path else None,
        "image_partition_counts": dict(sorted(Counter(x["split"] for x in images).items())),
        "set_count": len(result),
        "sets": result,
    }


def parse_counts(items):
    counts = {}
    for item in items:
        if item.count("=") != 1:
            raise ValueError("Use partition=count, e.g. train=30")
        name, value = item.split("=")
        if not name or name in counts:
            raise ValueError("Duplicate or empty partition name")
        counts[name] = int(value)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--rules", type=Path, default=Path(__file__).with_name("rules.json"))
    p.add_argument("--partitions", type=Path,
                   help="Filename-to-partition mapping, or image_partitions object")
    p.add_argument("--counts", nargs="+", required=True, metavar="PARTITION=N",
                   help="Sets PER RULE in each partition")
    p.add_argument("--images-per-set", type=int, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        manifest = make_manifest(args.data_dir, args.rules, args.partitions,
                                 parse_counts(args.counts), args.images_per_set,
                                 args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists() and read_json(args.out) != manifest:
            raise ValueError("Existing output differs: choose another manifest path")
        args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Validated {manifest['set_count']} identifiable sets -> {args.out}")
    except (ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
