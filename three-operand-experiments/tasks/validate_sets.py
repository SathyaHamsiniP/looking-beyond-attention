"""Validate manifests against original metadata and actual image files."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from .build_sets import CHECKSUM, VERSION_NAMES, split_records, validate_item
    from .rules import EXPRESSIONS, library, truth_tables
except ImportError:
    from build_sets import CHECKSUM, VERSION_NAMES, split_records, validate_item
    from rules import EXPRESSIONS, library, truth_tables


def validate(manifest, data_dir):
    root = Path(data_dir).resolve()
    raw = (root / "metadata.json").read_bytes()
    records = json.loads(raw)["records"]
    if manifest["task_version"] not in VERSION_NAMES:
        raise ValueError("Unknown task version")
    if {p.name for p in (root / "images").glob("*.png")} != {
        r["filename"] for r in records
    }:
        raise ValueError("Expected exactly the dataset metadata image files")
    rules = library(manifest["task_version"])
    if manifest["rules"] != {r: EXPRESSIONS[r] for r in rules} or manifest[
        "truth_tables"
    ] != truth_tables(rules):
        raise ValueError("Rule library differs from task version")
    if len({tuple(x) for x in manifest["truth_tables"].values()}) != len(rules):
        raise ValueError("Equivalent rules")
    if (
        manifest["metadata_sha256"] != hashlib.sha256(raw).hexdigest()
        or manifest["dataset_sha256"] != CHECKSUM
    ):
        raise ValueError("Dataset fingerprint mismatch")
    split = split_records(records, manifest["seed"])
    if split != manifest["split_manifest"]:
        raise ValueError("Split differs from deterministic split")
    by_name = {r["filename"]: r for r in records}
    ids = set()
    counts = Counter(
        (s["partition"], o["filename"])
        for s in manifest["sets"]
        for o in s["observations"]
    )
    distribution = Counter()
    for item in manifest["sets"]:
        if item["set_id"] in ids:
            raise ValueError("Duplicate set ID")
        ids.add(item["set_id"])
        report = validate_item(item, rules, manifest["task_version"])
        if (
            item["validation"] != report
            or len(item["observations"]) != item["examples_per_set"]
        ):
            raise ValueError("Stale validation report or size")
        distribution[item["partition"], item["rule"]] += 1
        for obs in item["observations"]:
            if (
                obs["bits"] != by_name[obs["filename"]]["bits"]
                or split[obs["filename"]] != item["partition"]
            ):
                raise ValueError("Metadata mismatch or partition leakage")
            path = (root / obs["path"]).resolve()
            if path != root / "images" / obs["filename"] or not path.is_file():
                raise ValueError("Missing image or unsafe path")
        reuse = any(
            counts[item["partition"], o["filename"]] > 1 for o in item["observations"]
        )
        if reuse != item["images_reused_across_sets"]:
            raise ValueError("Incorrect reuse flag")
    if (
        not ids
        or len(distribution) != 3 * len(rules)
        or len(set(distribution.values())) != 1
    ):
        raise ValueError("Rules must be balanced in all partitions")
    return {
        "sets": len(ids),
        "partitions": dict(Counter(split.values())),
        "oracle_accuracy": 1.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(validate(json.loads(a.manifest.read_text()), a.data_dir), indent=2)
    )


if __name__ == "__main__":
    main()
