"""Score the same held-out image sets before or after LoRA."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from build_training_sets import load
from common import (
    model_args,
    settings,
    load_model,
    inputs,
    score,
    run_folder,
    rows,
    append,
    save,
    adapter_identity,
    digest,
    memory,
    mapping,
)


def summarize(values):
    values = list(values)
    groups = defaultdict(list)
    for row in values:
        groups[row["true_rule"]].append(row["correct"])
    return dict(
        trials=len(values),
        sets=len({r["set_id"] for r in values}),
        accuracy=sum(r["correct"] for r in values) / len(values),
        per_rule={k: sum(v) / len(v) for k, v in groups.items()},
        by_rotation={
            str(i): sum(r["correct"] for r in values if r["rotation"] == i)
            / sum(r["rotation"] == i for r in values)
            for i in range(3)
        },
        predicted_rules=dict(Counter(r["prediction"] for r in values)),
        answer_labels=dict(Counter(r["answer_label"] for r in values)),
        confusion={
            truth: dict(
                Counter(r["prediction"] for r in values if r["true_rule"] == truth)
            )
            for truth in groups
        },
        chance=1 / 3,
        caution="Sets reuse images within each partition; rotations are repeated measurements, not independent examples.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_args(p)
    p.add_argument("--partition", choices=("validation", "test"), default="validation")
    a = p.parse_args()
    manifest = load(a.manifest, a.data_dir)
    config = settings(a)
    if a.adapter and json.loads((a.adapter / "experiment.json").read_text())[
        "manifest_hash"
    ] != digest(manifest):
        raise ValueError("Adapter was trained on a different manifest")
    selected = [s for s in manifest["sets"] if s["partition"] == a.partition]
    if a.smoke_test:
        selected = selected[:2]
    expected = {f"{s['set_id']}:{r}" for s in selected for r in range(3)}
    if a.validate_only:
        print("Checked", len(expected), "evaluation trials; no model loaded")
        return
    out = run_folder(
        a.output_root,
        dict(
            settings=config,
            kind="evaluation",
            partition=a.partition,
            smoke_test=a.smoke_test,
            adapter=adapter_identity(a.adapter),
        ),
        manifest,
    )
    path = out / "predictions.jsonl"
    done = rows(path)
    if not set(done) <= expected:
        raise ValueError("Unexpected saved trials")
    adapter = load_model(a)
    for item in selected:
        for rotation in range(3):
            key = f"{item['set_id']}:{rotation}"
            if key in done:
                continue
            prob = score(adapter, inputs(adapter, item, a.data_dir, rotation), rotation)
            prediction = max(prob, key=prob.get)
            row = dict(
                set_id=item["set_id"],
                true_rule=item["rule"],
                image_files=item["image_files"],
                rotation=rotation,
                mapping=mapping(rotation),
                prediction=prediction,
                answer_label=next(
                    l for l, r in mapping(rotation).items() if r == prediction
                ),
                probabilities=prob,
                correct=prediction == item["rule"],
            )
            append(path, key, row)
            done[key] = row
    if set(done) != expected:
        raise ValueError("Incomplete evaluation")
    save(out / "metrics.json", summarize(done.values()))
    memory()
    print(json.dumps(summarize(done.values()), indent=2))


if __name__ == "__main__":
    main()
