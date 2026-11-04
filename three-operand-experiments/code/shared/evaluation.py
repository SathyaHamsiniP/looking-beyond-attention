"""Matched variable-option baseline and adapter evaluation."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data_loader import load_manifest, option_labels, prompt, rotated, shortcut_control
from .model_qwen35 import load, model_metadata, score
from .run_utils import (
    Journal,
    atomic_json,
    dataset_fingerprint,
    file_fingerprint,
    prepare_run,
)


def metrics(rows):
    rows = list(rows)
    measured = [r for r in rows if r["control"] == "intact"]
    if not measured:
        raise ValueError("No measured predictions")
    by_rule = defaultdict(list)
    by_set = defaultdict(list)
    for r in measured:
        by_rule[r["truth"]].append(r["correct"])
        by_set[r["set_id"]].append(r)
    counts = Counter(r["truth"] for r in measured)
    return dict(
        n=len(measured),
        accuracy=sum(r["correct"] for r in measured) / len(measured),
        chance=sum(1 / len(r["probabilities"]) for r in measured) / len(measured),
        majority_class_baseline=max(counts.values()) / len(measured),
        per_rule_accuracy={k: sum(v) / len(v) for k, v in by_rule.items()},
        class_counts=dict(counts),
        all_orders_correct=sum(all(r["correct"] for r in rs) for rs in by_set.values())
        / len(by_set),
        prediction_agreement_across_orders=sum(
            len({r["prediction"] for r in rs}) == 1 for rs in by_set.values()
        )
        / len(by_set),
        shortcut_control="Common observations only: ambiguous; report probabilities and candidate mass, no accuracy.",
    )


def add_args(p):
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--partition", choices=("validation", "test"), default="validation")
    p.add_argument("--revision", default="main")
    p.add_argument(
        "--precision", choices=("4bit", "fp16", "bf16", "fp32"), default="4bit"
    )
    p.add_argument("--adapter", type=Path)
    p.add_argument("--rotations", type=int, default=0, help="0 = all cyclic orders")
    p.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--symbolic",
        action="store_true",
        help="Text configurations control; not visual inference",
    )
    p.add_argument("--shortcut-control", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--validate-only", action="store_true")


def main(experiment=1):
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    a = p.parse_args()
    m = load_manifest(a.manifest, a.data_dir)
    selected = [s for s in m["sets"] if s["partition"] == a.partition]
    if a.rotations < 0 or a.rotations > len(m["rules"]):
        p.error("Invalid rotation count")
    if a.smoke_test:
        selected = selected[:2]
    if a.validate_only:
        print(f"Validated {len(selected)} {a.partition} sets; no inference")
        return
    random.seed(a.seed)
    import torch

    torch.manual_seed(a.seed)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("validate_only", "results_root")
    }
    config["data_fingerprint"] = dataset_fingerprint(m, a.data_dir)
    if a.adapter:
        config["adapter_hashes"] = {
            f.name: file_fingerprint(f)
            for f in sorted(a.adapter.glob("*"))
            if f.is_file()
        }
    if a.adapter:
        from .model_qwen35 import validate_adapter

        adapter_info = validate_adapter(a.adapter, m)
        if adapter_info["revision"] != a.revision:
            raise ValueError(
                "Adapter revision selector differs; use the training revision"
            )
    adapter = load(a.precision, a.revision, a.adapter, a.max_pixels)
    if a.adapter and adapter_info["resolved_revision"] != getattr(
        adapter.model.config, "_commit_hash", None
    ):
        raise ValueError("Resolved base revision differs from adapter training")
    metadata = model_metadata(adapter, option_labels(len(m["rules"])))
    config["model"] = metadata
    out = prepare_run(a.results_root, experiment, m, config)
    journal = Journal(out / "predictions.jsonl")
    atomic_json(out / "model.json", metadata)
    expected = {
        f"{item['set_id']}:{rotation}:{control}"
        for item in selected
        for rotation in range(a.rotations or len(m["rules"]))
        for control in (
            ["intact", "uninformative_only"] if a.shortcut_control else ["intact"]
        )
    }
    if not set(journal.rows).issubset(expected):
        raise ValueError("Unexpected saved predictions")
    for item in selected:
        for rotation in range(a.rotations or len(m["rules"])):
            scene = rotated(item, rotation)
            controls = [("intact", scene, [scene["rule"]])]
            if a.shortcut_control:
                changed, candidates = shortcut_control(scene, m)
                controls.append(("uninformative_only", changed, candidates))
            for control, scene, candidates in controls:
                key = f"{item['set_id']}:{rotation}:{control}"
                if key in journal.rows:
                    continue
                paths, text, mapping = prompt(scene, m, a.data_dir, a.symbolic)
                probs = score(adapter, adapter._inputs(paths, text), mapping)
                prediction = max(probs, key=probs.get)
                journal.add(
                    key,
                    dict(
                        set_id=item["set_id"],
                        rotation=rotation,
                        partition=a.partition,
                        control=control,
                        truth=item["rule"],
                        prediction=prediction,
                        correct=prediction == item["rule"]
                        if control == "intact"
                        else None,
                        probabilities=probs,
                        option_mapping=mapping,
                        candidates=candidates,
                        candidate_probability_mass=sum(probs[r] for r in candidates),
                    ),
                )
    if set(journal.rows) != expected:
        raise ValueError("Incomplete evaluation")
    atomic_json(out / "metrics.json", metrics(journal.rows.values()))
    (out / "summary.md").write_text(
        "Matched option evaluation completed. See metrics.json. Accuracy alone does not establish logical reasoning.\n"
    )
    print(json.dumps(metrics(journal.rows.values()), indent=2))
