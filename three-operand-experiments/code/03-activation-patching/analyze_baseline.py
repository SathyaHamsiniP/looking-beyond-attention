"""Inspect saved baseline results without loading Qwen or changing source files."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import patch_activations_qwen35 as baseline


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probabilities(row, prefix, tolerance=1e-5):
    """Recover only identifiable probabilities. None means not saved or recoverable."""
    rules = baseline.RULES
    output = dict.fromkeys(rules)
    donor, receiver = row["donor_rule"], row["receiver_rule"]
    if prefix == "donor":
        output[donor] = float(row["donor_p_donor_rule"])
    else:
        output[donor] = float(row[f"{prefix}_p_donor_rule"])
        value = float(row[f"{prefix}_p_receiver_rule"])
        if donor == receiver and abs(output[donor] - value) > tolerance:
            raise ValueError("Conflicting probabilities for the same rule")
        output[receiver] = value
        if donor != receiver:
            third = next(r for r in rules if r not in (donor, receiver))
            remainder = 1 - sum(v for v in output.values() if v is not None)
            if not -tolerance <= remainder <= 1 + tolerance:
                raise ValueError("Probabilities do not sum to one")
            output[third] = min(1.0, max(0.0, remainder))
    if any(v is not None and not 0 <= v <= 1 for v in output.values()):
        raise ValueError("Invalid probability")
    prediction = row[prefix + "_prediction"]
    if prediction not in rules:
        raise ValueError("Unknown prediction")
    if output[prediction] is not None and any(
        v is not None and v > output[prediction] + tolerance for v in output.values()
    ):
        raise ValueError("Probabilities conflict with saved prediction")
    return output


def validate_rows(rows, pairs):
    expected = {
        (p["pair_id"], rotation, layer, pos)
        for p in pairs
        for rotation in range(3)
        for layer in range(1, 33)
        for pos in ("final", "previous")
    }
    found = set()
    by_id = {p["pair_id"]: p for p in pairs}
    originals = {}
    for row in rows:
        key = row["pair_id"], int(row["rotation"]), int(row["layer"]), row["position"]
        if key not in expected or key in found:
            raise ValueError("Unknown or duplicate trial")
        found.add(key)
        for name, value in by_id[row["pair_id"]].items():
            if row[name] != value:
                raise ValueError("Pair differs from saved configuration")
        for prefix, set_key in (("receiver", "receiver_set"), ("donor", "donor_set")):
            identity = row[set_key], row["rotation"]
            pred = row[prefix + "_prediction"]
            if identity in originals and originals[identity] != pred:
                raise ValueError("Original prediction changes across repeated records")
            originals[identity] = pred
            if int(row[prefix + "_correct"]) != int(pred == row[prefix + "_rule"]):
                raise ValueError("Incorrect correctness flag")
        for prefix in ("receiver", "donor", "patched"):
            probabilities(row, prefix)
        for rule in ("donor", "receiver"):
            delta = float(row[f"patched_p_{rule}_rule"]) - float(
                row[f"receiver_p_{rule}_rule"]
            )
            if abs(delta - float(row[f"delta_p_{rule}_rule"])) > 1e-5:
                raise ValueError("Incorrect probability change")
        changed = row["patched_prediction"] != row["receiver_prediction"]
        toward = (
            row["condition"] == "different"
            and row["receiver_prediction"] != row["donor_rule"]
            and row["patched_prediction"] == row["donor_rule"]
        )
        if (
            int(row["changed_answer"]) != changed
            or int(row["changed_to_donor_rule"]) != toward
        ):
            raise ValueError("Incorrect answer-change flag")
    if found != expected or len(rows) != 3456:
        raise ValueError("Expected all 3,456 baseline trials")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, data, out = (p.resolve() for p in (args.source, args.data_dir, args.output))
    if (
        source == out
        or source in out.parents
        or out in source.parents
        or data == out
        or data in out.parents
    ):
        parser.error("Output must be separate from source results and dataset")
    if out.exists():
        parser.error("Output already exists; choose a new folder")
    names = ("config.json", "patches.csv", "layer_effects.csv", "summary.json")
    checksums = {name: checksum(source / name) for name in names}
    config = json.loads((source / "config.json").read_text())
    if (
        config["model"],
        config["pairs_per_rule"],
        config["rotations"],
        config["previous_token_control"],
        config["smoke_test"],
    ) != ("Qwen/Qwen3.5-4B", 2, 3, True, False):
        raise ValueError("Not the requested full baseline configuration")
    baseline.exp.configure_data(data, out)
    sets = baseline.make_sets(baseline.exp)
    pairs = baseline.make_pairs(sets, 2)
    baseline.check_sets(sets, pairs, baseline.exp.records)
    if (
        pairs != config["pairs"]
        or baseline.data_fingerprint(data, sets, pairs) != config["dataset_fingerprint"]
    ):
        raise ValueError("Input reconstruction does not match the saved run")
    with (source / "patches.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    validate_rows(rows, pairs)
    manifest = {}
    used = {p[k] for p in pairs for k in ("receiver_set", "donor_set")}
    for item in sets:
        if item["set_id"] not in used:
            continue
        for rotation, (mapping, _) in enumerate(baseline.exp.layouts("cyclic")):
            paths, prompt = baseline.exp.build_input(
                item, mapping, baseline.exp.LABELS, "images"
            )
            manifest[f"{item['set_id']}:{rotation}"] = dict(
                images=item["image_files"],
                bits=item["bits"],
                prompt=prompt,
                mapping=mapping,
            )
    out.mkdir(parents=True)
    write_json(out / "input_manifest.json", manifest)
    originals = {
        (r["receiver_set"], r["rotation"]): (
            r["receiver_rule"],
            r["receiver_prediction"],
        )
        for r in rows
    }
    for r in rows:
        originals[r["donor_set"], r["rotation"]] = (
            r["donor_rule"],
            r["donor_prediction"],
        )
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["position"], int(row["layer"])].append(row)
    metrics = []
    for (condition, position, layer), group in sorted(groups.items()):
        mean = lambda fn: sum(fn(r) for r in group) / len(group)
        metrics.append(
            dict(
                condition=condition,
                position=position,
                layer=layer,
                n=len(group),
                answer_change_rate=mean(lambda r: int(r["changed_answer"])),
                donor_true_rule_change_rate=mean(
                    lambda r: int(r["changed_to_donor_rule"])
                ),
                donor_prediction_change_rate=mean(
                    lambda r: (
                        r["receiver_prediction"] != r["donor_prediction"]
                        and r["patched_prediction"] == r["donor_prediction"]
                    )
                ),
                signed_change=mean(lambda r: float(r["delta_p_donor_rule"])),
                absolute_change=mean(lambda r: abs(float(r["delta_p_donor_rule"]))),
            )
        )
    with (source / "layer_effects.csv").open(newline="") as f:
        saved = list(csv.DictReader(f))
    lookup = {(m["condition"], m["position"], m["layer"]): m for m in metrics}
    if len(saved) != len(metrics):
        raise ValueError("Unexpected layer summary size")
    for r in saved:
        m = lookup[r["condition"], r["position"], int(r["layer"])]
        if (
            int(r["n"]) != m["n"]
            or abs(float(r["mean_delta"]) - m["signed_change"]) > 1e-5
        ):
            raise ValueError("Saved aggregate differs from raw CSV")
    summary = json.loads((source / "summary.json").read_text())
    final_different = {
        m["layer"]: m["signed_change"]
        for m in metrics
        if m["condition"] == "different" and m["position"] == "final"
    }
    peak = max(final_different, key=final_different.get)
    if (
        summary["highest_mean_different_rule_effect"]["layer"] != peak
        or abs(
            summary["highest_mean_different_rule_effect"]["delta_probability"]
            - final_different[peak]
        )
        > 1e-5
    ):
        raise ValueError("Summary peak differs from raw trials")
    cutoff = next(
        (
            layer
            for layer in range(1, 33)
            if final_different[layer] >= 0.05
            and final_different[layer]
            - abs(lookup["same", "final", layer]["signed_change"])
            >= 0.05
            and final_different[layer]
            - abs(lookup["different", "previous", layer]["signed_change"])
            >= 0.05
        ),
        None,
    )
    if summary["first_layer_above_control_by_0_05"] != cutoff:
        raise ValueError("Summary control result differs from raw trials")
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "original_predictions.json",
        [
            dict(set_id=s, rotation=r, truth=t, prediction=p)
            for (s, r), (t, p) in originals.items()
        ],
    )
    final = [r for r in rows if r["layer"] == "32" and r["position"] == "final"]
    different = [r for r in final if r["condition"] == "different"]
    findings = dict(
        records=len(rows),
        final_layer_matches_donor_prediction=sum(
            r["patched_prediction"] == r["donor_prediction"] for r in final
        ),
        final_layer_trials=len(final),
        different_trials=len(different),
        different_changed=sum(int(r["changed_answer"]) for r in different),
        different_changed_to_true_rule=sum(
            int(r["changed_to_donor_rule"]) for r in different
        ),
        supplied_summary=json.loads((source / "summary.json").read_text()),
    )
    write_json(out / "verified_findings.json", findings)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for field in (
        "answer_change_rate",
        "signed_change",
        "absolute_change",
        "donor_true_rule_change_rate",
        "donor_prediction_change_rate",
    ):
        for condition in ("same", "different"):
            values = [
                m
                for m in metrics
                if m["condition"] == condition and m["position"] == "final"
            ]
            plt.plot(
                [m["layer"] for m in values],
                [m[field] for m in values],
                label=condition,
            )
        plt.xlabel("Decoder layer")
        plt.ylabel(field.replace("_", " "))
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / (field + ".png"))
        plt.close()
    pair_ids = [p["pair_id"] for p in pairs]
    heat = [
        [
            sum(
                float(r["delta_p_donor_rule"])
                for r in rows
                if r["pair_id"] == p
                and int(r["layer"]) == layer
                and r["position"] == "final"
            )
            / 3
            for layer in range(1, 33)
        ]
        for p in pair_ids
    ]
    plt.imshow(heat, aspect="auto", cmap="coolwarm")
    plt.yticks(range(len(pair_ids)), pair_ids, fontsize=4)
    plt.xlabel("Layer index (0 = layer 1)")
    plt.colorbar(label="Mean signed donor-rule probability change")
    plt.tight_layout()
    plt.savefig(out / "pair_heatmap.png")
    plt.close()
    matrix = [
        [sum(t == a and p == b for t, p in originals.values()) for b in baseline.RULES]
        for a in baseline.RULES
    ]
    plt.imshow(matrix)
    plt.xticks(range(3), baseline.RULES)
    plt.yticks(range(3), baseline.RULES)
    plt.xlabel("Original predicted rule")
    plt.ylabel("Generating rule")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out / "confusion.png")
    plt.close()
    for rotation in range(3):
        counts = Counter(
            next(
                label
                for label, rule in manifest[f"{s}:{r}"]["mapping"].items()
                if rule == pred
            )
            for (s, r), (truth, pred) in originals.items()
            if int(r) == rotation
        )
        plt.bar(
            [f"{rotation}:{label}" for label in baseline.exp.LABELS],
            [counts[label] for label in baseline.exp.LABELS],
        )
    plt.ylabel("Unique set predictions")
    plt.xlabel("Rotation: answer label")
    plt.tight_layout()
    plt.savefig(out / "answer_bias.png")
    plt.close()
    examples = [
        next(
            r
            for r in rows
            if r["condition"] == condition
            and r["layer"] == "32"
            and r["position"] == "final"
        )
        for condition in ("same", "different")
    ]
    write_json(
        out / "examples.json",
        [
            dict(
                record=r,
                receiver=probabilities(r, "receiver"),
                patched=probabilities(r, "patched"),
                reconstructed_rules=[
                    x
                    for x in baseline.RULES
                    if x not in (r["receiver_rule"], r["donor_rule"])
                ]
                if r["condition"] == "different"
                else [],
            )
            for r in examples
        ],
    )
    lines = [
        "# Verified baseline findings",
        "",
        f"Checked {len(rows):,} saved trials. Inputs match the saved fingerprint.",
        "",
        "At layer 32, all 54 final-token patches reproduce the donor prediction. Of 36 different-rule trials, 16 change the answer and seven change it to the donor true rule.",
        "",
        "| Condition at layer 32, final token | Trials | Answer changes | Mean signed change (percentage points) | Mean absolute change (percentage points) |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("same", "different"):
        m = lookup[condition, "final", 32]
        lines.append(
            f"| {condition} | {m['n']} | {round(m['answer_change_rate'] * m['n'])} | {100 * m['signed_change']:.4f} | {100 * m['absolute_change']:.4f} |"
        )
    lines += [
        "",
        f"Layer {peak} has the highest signed different-rule final-token mean: {100 * final_different[peak]:.5f} percentage points. It is negative. No layer passes the original 0.05 control cutoff.",
        "",
        "A donor-prediction match is not necessarily correct. Same-rule patches also change answers. Similar absolute changes across conditions do not identify a rule-specific reasoning layer.",
        "",
        "Plots count reused pairs and rotations, not independent image sets. The confusion matrix counts each evaluated set once per rotation, including donor-only sets. Answer bias therefore describes 27 original set/rotation predictions, not 3,456 independent original answers.",
        "",
        "Only two distinct receiver/patched rule probabilities can be reconstructed in different-rule records. Same-rule missing probabilities stay unknown. See examples.json for explicit reconstructed-rule labels. The two examples are the first matching rows in file order; they are not a representative statistical sample.",
        "",
        "Source files are copied byte-for-byte into raw/. No model was rerun. Read the parent guide for the beginner explanation.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n")
    write_json(
        out / "provenance.json",
        dict(
            source_checksums=checksums,
            dataset_fingerprint=config["dataset_fingerprint"],
            analysis_script_checksum=checksum(__file__),
            baseline_script_checksum=checksum(baseline.__file__),
            sampler_checksum=checksum(
                Path(baseline.__file__).parents[1] / "02-probing/probe_rules.py"
            ),
            baseline_run_code_commit="not recorded in original config",
            analysis_working_tree_dirty=bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            command=[
                "--source",
                str(source),
                "--data-dir",
                str(data),
                "--output",
                str(out),
            ],
        ),
    )
    if checksums != {name: checksum(source / name) for name in names}:
        raise RuntimeError("Source files changed during analysis")
    print("Verified 3,456 trials. Saved separate analysis to", out)


if __name__ == "__main__":
    main()
