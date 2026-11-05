"""Compare saved predictions on the same examples. No model is loaded."""

import argparse
import csv
import json
from pathlib import Path
from common import matched, rows, save
from evaluate_lora import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--finetuned", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    left = json.loads((a.baseline / "config.json").read_text())
    right = json.loads((a.finetuned / "config.json").read_text())
    matched(
        left, right, ("settings", "manifest_hash", "partition", "smoke_test", "kind")
    )
    if left["adapter"] or not right["adapter"]:
        raise ValueError("Supply pretrained baseline first, adapter evaluation second")
    before = rows(a.baseline / "predictions.jsonl")
    after = rows(a.finetuned / "predictions.jsonl")
    manifest = json.loads((a.baseline / "manifest.json").read_text())
    selected = [s for s in manifest["sets"] if s["partition"] == left["partition"]]
    if left["smoke_test"]:
        selected = selected[:2]
    expected = {f"{s['set_id']}:{r}" for s in selected for r in range(3)}
    if set(before) != expected or set(after) != expected:
        raise ValueError("Incomplete or different evaluation example IDs")
    output = []
    for key, b in before.items():
        f = after[key]
        matched(b, f, ("set_id", "true_rule", "image_files", "rotation", "mapping"))
        output.append(
            dict(
                key=key,
                true_rule=b["true_rule"],
                rotation=b["rotation"],
                before=b["prediction"],
                after=f["prediction"],
                changed=b["prediction"] != f["prediction"],
                correct_before=b["correct"],
                correct_after=f["correct"],
                delta_true_rule_probability=f["probabilities"][b["true_rule"]]
                - b["probabilities"][b["true_rule"]],
                **{
                    "delta_" + r: f["probabilities"][r] - b["probabilities"][r]
                    for r in ("R1", "R2", "R3")
                },
            )
        )
    if a.output.exists():
        raise ValueError("Choose a fresh comparison output folder")
    a.output.mkdir(parents=True)
    with (a.output / "matched_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    first, second = summarize(before.values()), summarize(after.values())
    save(
        a.output / "comparison.json",
        dict(
            before=first,
            after=second,
            accuracy_difference=second["accuracy"] - first["accuracy"],
            changed=sum(r["changed"] for r in output),
            sources=[str(a.baseline), str(a.finetuned)],
        ),
    )
    import matplotlib.pyplot as plt

    for name, label in ((first, "pretrained"), (second, "LoRA")):
        plt.plot(
            list(name["per_rule"]),
            list(name["per_rule"].values()),
            marker="o",
            label=label,
        )
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(a.output / "per_rule_accuracy.png")
    plt.close()
    for report, label in ((first, "pretrained"), (second, "LoRA")):
        plt.bar(
            [label + ":" + l for l in "XYZ"],
            [report["answer_labels"].get(l, 0) for l in "XYZ"],
        )
    plt.ylabel("Predicted answer count")
    plt.tight_layout()
    plt.savefig(a.output / "answer_bias.png")
    plt.close()
    for report, label in ((first, "pretrained"), (second, "LoRA")):
        matrix = [
            [report["confusion"].get(t, {}).get(r, 0) for r in ("R1", "R2", "R3")]
            for t in ("R1", "R2", "R3")
        ]
        plt.imshow(matrix)
        plt.xticks(range(3), ("R1", "R2", "R3"))
        plt.yticks(range(3), ("R1", "R2", "R3"))
        plt.xlabel("Predicted")
        plt.ylabel("True rule")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(a.output / (label + "_confusion.png"))
        plt.close()
    (a.output / "README.md").write_text(
        "These are matched predictions on the same image sets and three answer orders. See comparison.json for counts, per-rule and per-rotation accuracy. Reused images and repeated answer orders are not independent samples. Higher accuracy does not prove Boolean reasoning.\n"
    )


if __name__ == "__main__":
    main()
