"""Compare complete matched patching runs, including every receiver."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04-lora"))
from common import matched, rows, save


def compare_configs(a, b):
    matched(a, b, ("settings", "manifest_hash", "plan", "smoke_test", "kind"))
    if a["adapter"] or not b["adapter"]:
        raise ValueError("Need pretrained run first and LoRA run second")


def aggregate(values):
    groups = defaultdict(list)
    for row in values:
        groups[row["condition"], row["position"], row["layer"]].append(row)
    out = []
    for (condition, position, layer), group in sorted(groups.items()):
        mean = lambda key: sum(r[key] for r in group) / len(group)
        out.append(
            dict(
                condition=condition,
                position=position,
                layer=layer,
                n=len(group),
                signed_donor=mean("delta_donor"),
                absolute_donor=sum(abs(r["delta_donor"]) for r in group) / len(group),
                receiver_change=mean("delta_receiver"),
                answer_change=mean("changed_answer"),
                donor_true_transfer=mean("changed_to_donor_true"),
                donor_prediction_transfer=mean("changed_to_donor_prediction"),
            )
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--finetuned", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    before = json.loads((a.baseline / "config.json").read_text())
    after = json.loads((a.finetuned / "config.json").read_text())
    compare_configs(before, after)
    left = rows(a.baseline / "patches.jsonl")
    right = rows(a.finetuned / "patches.jsonl")
    plan = before["plan"]
    expected = {
        f"{p['pair_id']}:{r}:{l}:{s}"
        for p in plan["pairs"]
        for r in plan["rotations"]
        for l in plan["layers"]
        for s in plan["positions"]
    }
    if set(left) != expected or set(right) != expected:
        raise ValueError("Incomplete or different patching trials")
    for key, row in left.items():
        matched(
            row,
            right[key],
            (
                "pair_id",
                "receiver",
                "donor",
                "receiver_rule",
                "donor_rule",
                "rotation",
                "mapping",
                "layer",
                "position",
                "shared_images",
            ),
        )
    if a.output.exists():
        raise ValueError("Choose a new comparison folder")
    a.output.mkdir(parents=True)
    first, second = aggregate(left.values()), aggregate(right.values())
    save(a.output / "before.json", first)
    save(a.output / "after.json", second)
    diffs = []
    for b, f in zip(first, second):
        diffs.append(
            {
                **{k: b[k] for k in ("condition", "position", "layer", "n")},
                **{
                    key: f[key] - b[key]
                    for key in (
                        "signed_donor",
                        "absolute_donor",
                        "receiver_change",
                        "answer_change",
                        "donor_true_transfer",
                        "donor_prediction_transfer",
                    )
                },
            }
        )
    with (a.output / "differences.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diffs[0]))
        w.writeheader()
        w.writerows(diffs)
    import matplotlib.pyplot as plt

    for metric in (
        "signed_donor",
        "absolute_donor",
        "receiver_change",
        "answer_change",
        "donor_true_transfer",
        "donor_prediction_transfer",
    ):
        for values, label, style in (
            (first, "pretrained", "--"),
            (second, "LoRA", "-"),
        ):
            for condition in ("same", "different"):
                for position in plan["positions"]:
                    selected = sorted(
                        [
                            r
                            for r in values
                            if r["condition"] == condition and r["position"] == position
                        ],
                        key=lambda r: r["layer"],
                    )
                    plt.plot(
                        [r["layer"] for r in selected],
                        [r[metric] for r in selected],
                        style,
                        label=f"{label} {condition} {position}",
                    )
        plt.xlabel("Decoder layer")
        plt.ylabel(metric.replace("_", " "))
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(a.output / (metric + ".png"))
        plt.close()
    save(
        a.output / "sources.json",
        dict(
            baseline=str(a.baseline),
            finetuned=str(a.finetuned),
            settings=before["settings"],
            plan=plan,
        ),
    )
    (a.output / "README.md").write_text(
        "Plots include all matched trials. Signed and absolute changes answer different questions. Donor true-rule transfer is different from copying the donor prediction, which may be wrong. Same-rule controls and previous-token controls are shown. No layer is called a reasoning layer based on the largest effect. Repeated sets and shared images limit statistical independence.\n"
    )


if __name__ == "__main__":
    main()
