"""Compare all matched trials; no selection on model success."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from .run_utils import atomic_json


def summarize(rows, out):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"], row["position"], row["layer"]].append(row)
    stats = []
    for (condition, position, layer), values in sorted(grouped.items()):
        stats.append(
            dict(
                condition=condition,
                position=position,
                layer=layer,
                n=len(values),
                mean_delta_receiver_probability=sum(
                    r["delta_receiver_probability"] for r in values
                )
                / len(values),
                mean_delta_donor_probability=sum(
                    r["delta_donor_probability"] for r in values
                )
                / len(values),
                answer_change_rate=sum(r["changed_answer"] for r in values)
                / len(values),
                receiver_accuracy=sum(r["receiver_correct"] for r in values)
                / len(values),
                donor_accuracy=sum(r["donor_correct"] for r in values) / len(values),
            )
        )
    with (out / "layer_effects.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    for condition in ("same", "different"):
        for position in ("final", "previous"):
            selected = sorted(
                [
                    r
                    for r in stats
                    if r["condition"] == condition and r["position"] == position
                ],
                key=lambda r: r["layer"],
            )
            if selected:
                plt.plot(
                    [r["layer"] for r in selected],
                    [r["mean_delta_receiver_probability"] for r in selected],
                    label=f"{condition}: {position}",
                )
    plt.xlabel("Decoder layer")
    plt.ylabel("Change in P(receiver true rule)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "layer_effects.png")
    plt.close()
    effects = [
        r for r in stats if r["condition"] == "different" and r["position"] == "final"
    ]
    effects.sort(key=lambda r: r["layer"])
    peak = max(effects, key=lambda r: abs(r["mean_delta_receiver_probability"]))
    adjacent = [
        dict(
            layer=b["layer"],
            delta=b["mean_delta_receiver_probability"]
            - a["mean_delta_receiver_probability"],
        )
        for a, b in zip(effects, effects[1:])
        if b["layer"] == a["layer"] + 1
    ]
    atomic_json(
        out / "summary.json",
        dict(
            trials=len(rows),
            maximum_absolute_effect=peak,
            adjacent_layer_changes=adjacent,
            interpretation="Receiver true-rule probability uses the same target definition in both conditions. Donor-rule changes have different target meanings and are secondary. Maximum effect is not a unique reasoning location. No significance threshold is applied.",
        ),
    )
    return stats


def compare(paths, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty comparison output directory")
    configs = [json.loads((p / "config.json").read_text()) for p in paths]
    if any("plan_hash" not in c for c in configs):
        raise ValueError(
            "Legacy Experiment 03 has no matched manifest plan. Keep its path unchanged; do not compare it to V3 as matched data. Run manifest baseline patching for this comparison."
        )
    if (
        len({c["plan_hash"] for c in configs}) != 1
        or len({c["data_fingerprint"] for c in configs}) != 1
    ):
        raise ValueError("Patching sets, layers, orders, precision, or images differ")
    if any(c.get("adapter") for c in configs[:1]):
        raise ValueError("First path must be the untuned baseline")
    if any(c["model"] != configs[0]["model"] for c in configs[1:]):
        raise ValueError("Model revisions or answer tokens differ")
    all_rows = [
        [json.loads(line) for line in (p / "patches.jsonl").read_text().splitlines()]
        for p in paths
    ]
    expected = {
        f"{pair['pair_id']}:{rotation}:{position}:{layer}"
        for pair in configs[0]["plan"]["pairs"]
        for rotation in range(configs[0]["plan"]["rotations"])
        for position in configs[0]["plan"]["positions"]
        for layer in configs[0]["plan"]["layers"]
    }
    if any(
        len(rows) != len(expected) or {r["key"] for r in rows} != expected
        for rows in all_rows
    ):
        raise ValueError("Missing, duplicate, or extra patch trials")
    out.mkdir(parents=True, exist_ok=True)
    combined = []
    accuracy = []
    baseline = {r["key"]: r for r in all_rows[0]}
    import matplotlib.pyplot as plt

    for path, rows in zip(paths, all_rows):
        name = str(path)
        sub = out / f"variant-{len(accuracy)}"
        sub.mkdir()
        stats = summarize(rows, sub)
        unique = {(r["receiver"], r["rotation"]): r["receiver_correct"] for r in rows}
        accuracy.append(
            dict(
                variant=name, n=len(unique), accuracy=sum(unique.values()) / len(unique)
            )
        )
        for r in rows:
            combined.append(
                dict(
                    variant=name,
                    key=r["key"],
                    layer=r["layer"],
                    condition=r["condition"],
                    position=r["position"],
                    delta_receiver_probability=r["delta_receiver_probability"],
                    difference_from_baseline=r["delta_receiver_probability"]
                    - baseline[r["key"]]["delta_receiver_probability"],
                )
            )
        selected = [
            r
            for r in stats
            if r["condition"] == "different" and r["position"] == "final"
        ]
        plt.plot(
            [r["layer"] for r in selected],
            [r["mean_delta_receiver_probability"] for r in selected],
            label=f"variant-{len(accuracy) - 1}",
        )
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Change in P(receiver rule)")
    plt.tight_layout()
    plt.savefig(out / "comparison.png")
    plt.close()
    with (out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined[0]))
        w.writeheader()
        w.writerows(combined)
    atomic_json(out / "accuracy.json", accuracy)
    atomic_json(out / "sources.json", [str(p) for p in paths])
    (out / "summary.md").write_text(
        "All matched receivers were included, regardless of correctness. Accuracy here covers the patching receiver subset; use full evaluation for final accuracy. See accuracy.json and comparison.csv. These effects do not prove Boolean reasoning.\n"
    )
