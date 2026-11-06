"""Audit Experiment 04 saved artifacts and derive CPU-only rotation statistics."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RULES = ("R1", "R2", "R3")


def read_rows(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate prediction keys")
    return {r["key"]: r for r in records}


def summary(records):
    records = list(records)
    groups = defaultdict(list)
    for r in records:
        groups[r["set_id"]].append(r)
        assert r["mapping"][r["answer_label"]] == r["prediction"]
        assert r["correct"] == (r["prediction"] == r["true_rule"])
        assert set(r["probabilities"]) == set(RULES)
        assert all(0 <= p <= 1 for p in r["probabilities"].values())
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
        assert r["prediction"] == max(r["probabilities"], key=r["probabilities"].get)
    for group in groups.values():
        assert sorted(r["rotation"] for r in group) == [0, 1, 2]
        assert len({tuple(r["image_files"]) for r in group}) == 1
        assert len({r["true_rule"] for r in group}) == 1
        for rule in RULES:
            assert {next(k for k, v in r["mapping"].items() if v == rule) for r in group} == set("XYZ")
    return dict(
        trials=len(records), sets=len(groups),
        accuracy=sum(r["correct"] for r in records) / len(records),
        per_rule={t: sum(r["correct"] for r in records if r["true_rule"] == t) /
                  sum(r["true_rule"] == t for r in records) for t in RULES if any(r["true_rule"] == t for r in records)},
        by_rotation={str(i): sum(r["correct"] for r in records if r["rotation"] == i) /
                     sum(r["rotation"] == i for r in records) for i in range(3)},
        predicted_rules=dict(Counter(r["prediction"] for r in records)),
        answer_labels=dict(Counter(r["answer_label"] for r in records)),
        confusion={t: dict(Counter(r["prediction"] for r in records if r["true_rule"] == t)) for t in RULES},
        consistent_sets=sum(len({r["prediction"] for r in g}) == 1 for g in groups.values()),
    )


def paired_bootstrap(before, after, seed=2026, draws=20000):
    """Resample matched image sets, retaining all three rotations per draw."""
    ids = sorted({r["set_id"] for r in before.values()})
    scores = np.array([[np.mean([r["correct"] for r in rows.values() if r["set_id"] == s])
                        for rows in (before, after)] for s in ids])
    sample = np.random.default_rng(seed).integers(0, len(ids), size=(draws, len(ids)))
    means = scores[sample].mean(axis=1)
    intervals = np.quantile(np.column_stack((means, means[:, 1] - means[:, 0])), [.025, .975], axis=0)
    return dict(seed=seed, draws=draws, method="paired set-clustered percentile bootstrap, 95%",
                pretrained=intervals[:, 0].tolist(), finetuned=intervals[:, 1].tolist(),
                difference=intervals[:, 2].tolist(),
                limitation="Shared images across sets remain dependent; these intervals account only for rotations within a set. Selection uncertainty is not included.")


def analyze(root):
    before, after = [read_rows(root / name / "predictions.jsonl") for name in ("pretrained", "finetuned")]
    manifest = json.loads((root / "rule_identification_manifest.json").read_text())
    assert manifest["seed"] == 2026
    assert Counter(manifest["image_partitions"].values()) == {"train": 800, "validation": 100, "test": 100}
    assert Counter(s["partition"] for s in manifest["sets"]) == {"train": 90, "validation": 30, "test": 30}
    for partition, count in (("train", 30), ("validation", 10), ("test", 10)):
        sets = [s for s in manifest["sets"] if s["partition"] == partition]
        assert Counter(s["rule"] for s in sets) == {r: count for r in RULES}
        for s in sets:
            assert len(s["image_files"]) == len(set(s["image_files"])) == 10
            assert all(manifest["image_partitions"][image] == partition for image in s["image_files"])
    selected = {s["set_id"]: s for s in manifest["sets"] if s["partition"] == "validation"}
    expected = {f"{s}:{i}" for s in selected for i in range(3)}
    assert set(before) == set(after) == expected
    for key, b in before.items():
        f = after[key]
        for field in ("key", "set_id", "true_rule", "rotation", "image_files", "mapping"):
            assert b[field] == f[field], (key, field)
        s = selected[b["set_id"]]
        assert b["image_files"] == s["image_files"]
        assert b["true_rule"] == s["rule"]
    reports = [summary(r.values()) for r in (before, after)]
    comparison = json.loads((root / "comparison/comparison.json").read_text())
    for name, report, side in zip(("pretrained", "finetuned"), reports, ("before", "after")):
        saved = json.loads((root / name / "metrics.json").read_text())
        for field, value in report.items():
            if field != "consistent_sets":
                assert saved[field] == comparison[side][field] == value, field
    assert comparison["accuracy_difference"] == reports[1]["accuracy"] - reports[0]["accuracy"]
    changes = Counter()
    with (root / "comparison/matched_predictions.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == len(before) and {r["key"] for r in table} == expected
    for row in table:
        b, f = before[row["key"]], after[row["key"]]
        for column, value in dict(true_rule=b["true_rule"], rotation=b["rotation"], before=b["prediction"],
                                  after=f["prediction"], changed=b["prediction"] != f["prediction"],
                                  correct_before=b["correct"], correct_after=f["correct"]).items():
            assert row[column] == str(value)
        for rule in RULES:
            assert abs(float(row["delta_" + rule]) - (f["probabilities"][rule] - b["probabilities"][rule])) < 1e-12
        assert abs(float(row["delta_true_rule_probability"]) - (f["probabilities"][b["true_rule"]] - b["probabilities"][b["true_rule"]])) < 1e-12
        if b["prediction"] != f["prediction"]:
            changes["correct_to_wrong" if b["correct"] else "wrong_to_correct" if f["correct"] else "wrong_to_different_wrong"] += 1
    assert comparison["changed"] == sum(changes.values())
    losses = list(read_rows(root / "training/losses.jsonl").values())
    complete = json.loads((root / "training/complete.json").read_text())
    selection = json.loads((root / "training/selection.json").read_text())
    assert [r["step"] for r in losses] == list(range(1, 47))
    assert all(r["optimizer_step_occurred"] and r["lora_b_matrices_changed"] == r["lora_b_matrices"] == 248 for r in losses)
    assert complete["optimizer_updates"] == complete["steps_with_lora_b_changes"] == complete["steps"] == 46
    assert complete["test_used"] is False
    best = min((r for r in losses if "validation_loss" in r), key=lambda r: r["validation_loss"])
    assert selection["step"] == best["step"] and selection["loss"] == best["validation_loss"] == complete["best_validation_loss"]
    configs = [json.loads((root / name / "config.json").read_text()) for name in ("pretrained", "finetuned", "training")]
    assert configs[0]["settings"] == configs[1]["settings"] == configs[2]["settings"]
    assert configs[0]["manifest_hash"] == configs[1]["manifest_hash"] == configs[2]["manifest_hash"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert configs[0]["partition"] == configs[1]["partition"] == "validation"
    assert configs[0]["adapter"] is None and configs[1]["adapter"]
    assert all(not c["smoke_test"] for c in configs)
    training = configs[2]
    for field, value in dict(rank=8, alpha=16, dropout=.05, learning_rate=.0002,
                             epochs=2, gradient_accumulation=4, batch_size=1,
                             initial_gradient_scale=8, max_pixels=100352, max_tokens=16000).items():
        assert training[field] == value
    assert (root / "model_revision.txt").read_text().strip() == configs[0]["settings"]["revision"]
    return dict(before=reports[0], after=reports[1], changes=dict(changes),
                bootstrap=paired_bootstrap(before, after),
                example=[dict(before=before[k], after=after[k]) for k in sorted(before)[:3]])


def loss_plot(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    losses = list(read_rows(root / "training/losses.jsonl").values())
    val = [r for r in losses if "validation_loss" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in losses], [r["training_loss"] for r in losses], label="Training (update batches)", alpha=.7)
    ax.plot([r["step"] for r in val], [r["validation_loss"] for r in val], "o-", label="Validation (three rotations)")
    ax.axvline(20, linestyle="--", color="gray", label="Selected checkpoint")
    ax.set(xlabel="Successful optimizer update", ylabel="Full-vocabulary answer-token loss")
    ax.legend(); fig.tight_layout(); fig.savefig(output); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.results)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    loss_plot(a.results, a.output / "loss_curve.png")
    print(json.dumps({k: v for k, v in report.items() if k != "example"}, indent=2))


if __name__ == "__main__":
    main()
