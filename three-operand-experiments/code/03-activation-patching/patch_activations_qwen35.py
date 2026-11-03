#!/usr/bin/env python3
"""Patch Qwen3.5-4B decoder activations between held-out ten-image sets.

Same-rule donors are the control; different-rule donors test whether the
donor's rule becomes more likely. No model weights are changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "01-operator-controls"))
sys.path.insert(0, str(CODE_ROOT / "02-probing"))

import operator_controls_qwen35 as exp
from probe_concepts import decoder_layers
from probe_rules import make_sets, split_report
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root

RULES = ("R1", "R2", "R3")
THRESHOLD = 0.05  # Heuristic probability-point cutoff, not a significance test.
FIELDS = (
    "pair_id", "condition", "receiver_set", "receiver_rule",
    "donor_set", "donor_rule", "rotation", "layer", "position",
    "receiver_prediction", "donor_prediction", "patched_prediction",
    "receiver_correct", "donor_correct",
    "receiver_p_donor_rule", "donor_p_donor_rule", "patched_p_donor_rule",
    "delta_p_donor_rule", "receiver_p_receiver_rule",
    "patched_p_receiver_rule", "delta_p_receiver_rule",
    "changed_answer", "changed_to_donor_rule",
)


def make_pairs(sets: list[dict], pairs_per_rule: int) -> list[dict]:
    """Each receiver gets one same-rule donor and both other-rule donors."""
    test = [item for item in sets if item["split"] == "test"]
    by_rule = {
        rule: sorted((s for s in test if s["rule"] == rule),
                     key=lambda item: item["set_id"])
        for rule in RULES
    }
    if any(len(by_rule[rule]) != 5 for rule in RULES):
        raise ValueError("Expected five held-out ten-image sets per rule.")
    if not 1 <= pairs_per_rule <= 5:
        raise ValueError("--pairs-per-rule must be between 1 and 5.")

    pairs = []
    for receiver_rule in RULES:
        for index in range(pairs_per_rule):
            receiver = by_rule[receiver_rule][index]
            donors = [
                ("same", by_rule[receiver_rule][(index + 1) % 5]),
                *[
                    ("different", by_rule[rule][index])
                    for rule in RULES if rule != receiver_rule
                ],
            ]
            for condition, donor in donors:
                assert receiver["set_id"] != donor["set_id"]
                assert set(receiver["image_files"]).isdisjoint(donor["image_files"])
                pairs.append({
                    "pair_id": f'{receiver["set_id"]}__{condition}__{donor["set_id"]}',
                    "condition": condition,
                    "receiver_set": receiver["set_id"],
                    "receiver_rule": receiver_rule,
                    "donor_set": donor["set_id"],
                    "donor_rule": donor["rule"],
                })
    return pairs


def check_sets(sets: list[dict], pairs: list[dict], records: list[dict]) -> dict:
    test = [s for s in sets if s["split"] == "test"]
    train_names = {r["filename"] for r in records if r["split"] == "train"}
    test_names = {r["filename"] for r in records if r["split"] == "test"}
    assert len(records) == 1000 and len(train_names) == 800 and len(test_names) == 200
    assert train_names.isdisjoint(test_names)
    used = [name for item in test for name in item["image_files"]]
    assert len(test) == 15 and len(used) == len(set(used)) == 150
    assert set(used).issubset(test_names)
    for pair in pairs:
        assert pair["receiver_set"] != pair["donor_set"]
        assert (pair["receiver_rule"] == pair["donor_rule"]) == (pair["condition"] == "same")
    return {
        "dataset": "dataset-v1",
        "test_sets": len(test),
        "different_images_across_test_sets": len(used),
        "pairs": len(pairs),
        "same_rule_pairs": sum(p["condition"] == "same" for p in pairs),
        "different_rule_pairs": sum(p["condition"] == "different" for p in pairs),
        "held_out_only": True,
    }


def data_fingerprint(data_dir: Path, sets: list[dict], selected: list[dict]) -> str:
    names = set()
    by_id = {item["set_id"]: item for item in sets}
    for pair in selected:
        for key in ("receiver_set", "donor_set"):
            names.update(by_id[pair[key]]["image_files"])
    digest = hashlib.sha256((data_dir / "metadata.json").read_bytes())
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update((data_dir / "images" / name).read_bytes())
    return digest.hexdigest()


def get_inputs(adapter, item: dict, mapping: dict):
    paths, prompt = exp.build_input(item, mapping, exp.LABELS, "images")
    return adapter._inputs(paths, prompt)


def probabilities(model, inputs, mapping: dict) -> dict[str, float]:
    import torch
    with torch.inference_mode():
        result = model(**inputs, logits_to_keep=1, use_cache=False)
        logits = result.logits[0, -1].float()
        token_scores = torch.stack([
            logits[exp.TOKEN_IDS[label]] for label in exp.LABELS
        ])
        values = torch.softmax(token_scores, dim=0).cpu().tolist()
    output = {mapping[label]: float(p) for label, p in zip(exp.LABELS, values)}
    if not math.isclose(sum(output.values()), 1.0, abs_tol=1e-4):
        raise ValueError("Invalid answer probabilities.")
    return output


def record_original(adapter, inputs, mapping: dict, layers) -> dict:
    """One forward pass saves the last two token vectors from every layer."""
    import torch

    saved = {}
    handles = []

    def capture(layer_number):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 2:
                raise ValueError("Unexpected decoder output shape.")
            saved[layer_number] = {
                "final": hidden[0, -1].detach().cpu().clone(),
                "previous": hidden[0, -2].detach().cpu().clone(),
            }
        return hook

    try:
        for number, block in enumerate(layers, start=1):
            handles.append(block.register_forward_hook(capture(number)))
        prob = probabilities(adapter.model, inputs, mapping)
    finally:
        for handle in handles:
            handle.remove()
    if len(saved) != len(layers):
        raise RuntimeError("Did not capture all decoder layers.")
    return {"probabilities": prob, "activations": saved,
            "prediction": max(RULES, key=prob.get)}


def patched_probabilities(adapter, inputs, mapping: dict, block, donor_vector,
                          position: str) -> dict[str, float]:
    """Replace one token's decoder-layer output, without changing other tokens."""
    token_index = -1 if position == "final" else -2

    def replace(_module, _args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 2:
            raise ValueError("Unexpected decoder output shape.")
        vector = donor_vector.to(device=hidden.device, dtype=hidden.dtype)
        if vector.shape != hidden[0, token_index].shape:
            raise ValueError("Donor and receiver hidden dimensions differ.")
        changed = hidden.clone()
        changed[:, token_index, :] = vector
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        return changed

    handle = block.register_forward_hook(replace)
    try:
        return probabilities(adapter.model, inputs, mapping)
    finally:
        handle.remove()


def to_row(pair, rotation: int, layer: int, position: str,
           receiver: dict, donor: dict, patched: dict) -> dict:
    target = pair["donor_rule"]
    source = pair["receiver_rule"]
    before = receiver["probabilities"]
    after = patched
    pred = max(RULES, key=after.get)
    return {
        **pair, "rotation": rotation, "layer": layer, "position": position,
        "receiver_prediction": receiver["prediction"],
        "donor_prediction": donor["prediction"],
        "patched_prediction": pred,
        "receiver_correct": int(receiver["prediction"] == source),
        "donor_correct": int(donor["prediction"] == target),
        "receiver_p_donor_rule": before[target],
        "donor_p_donor_rule": donor["probabilities"][target],
        "patched_p_donor_rule": after[target],
        "delta_p_donor_rule": after[target] - before[target],
        "receiver_p_receiver_rule": before[source],
        "patched_p_receiver_rule": after[source],
        "delta_p_receiver_rule": after[source] - before[source],
        "changed_answer": int(pred != receiver["prediction"]),
        "changed_to_donor_rule": int(
            pair["condition"] == "different" and
            receiver["prediction"] != target and pred == target
        ),
    }


def write_json(path: Path, obj) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def saved_rows(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(rows[0]) != set(FIELDS):
        raise ValueError(f"Unexpected saved columns in {path}.")
    keys = {}
    for row in rows:
        key = row["pair_id"], int(row["rotation"]), int(row["layer"]), row["position"]
        if key in keys:
            raise ValueError(f"Duplicate saved result: {key}")
        keys[key] = row
    return keys


def summarize(rows_path: Path, out: Path, layer_count: int, smoke: bool,
              previous_control: bool) -> None:
    import matplotlib.pyplot as plt

    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["position"], int(row["layer"]))
        grouped[key].append(float(row["delta_p_donor_rule"]))
    aggregate = []
    for (condition, position, layer), values in sorted(grouped.items()):
        aggregate.append({
            "condition": condition, "position": position, "layer": layer,
            "n": len(values), "mean_delta": float(np.mean(values)),
            "std_delta": float(np.std(values)),
        })
    with (out / "layer_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    for condition in ("different", "same"):
        for position in ("final", "previous"):
            subset = [
                a for a in aggregate
                if a["condition"] == condition and a["position"] == position
            ]
            if not subset:
                continue
            subset.sort(key=lambda a: a["layer"])
            style = "-" if position == "final" else "--"
            plt.plot([a["layer"] for a in subset],
                     [a["mean_delta"] for a in subset],
                     style, label=f"{condition}: {position}")
    plt.axhline(0, color="grey", linewidth=1)
    plt.xlabel("Decoder layer output (1-based)")
    plt.ylabel("Change in P(donor rule)")
    plt.title("Activation patching — Qwen3.5-4B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "layer_effects.png", dpi=160)
    plt.close()

    final_different = {
        item["layer"]: item["mean_delta"] for item in aggregate
        if item["condition"] == "different" and item["position"] == "final"
    }
    final_same = {
        item["layer"]: item["mean_delta"] for item in aggregate
        if item["condition"] == "same" and item["position"] == "final"
    }
    previous = {
        item["layer"]: item["mean_delta"] for item in aggregate
        if item["condition"] == "different" and item["position"] == "previous"
    }
    if not final_different:
        raise ValueError("Missing different-rule final-token results.")
    peak = max(final_different, key=final_different.get)
    layers_tested = sorted(final_different)
    biggest_step = (
        max(layers_tested[1:], key=lambda l: final_different[l] - final_different[l - 1])
        if layers_tested == list(range(1, layer_count + 1))
        else None
    )
    first_over_control = next((
        layer for layer in layers_tested
        if final_different[layer] >= THRESHOLD
        and final_different[layer] - abs(final_same[layer]) >= THRESHOLD
        and (not previous_control
             or final_different[layer] - abs(previous[layer]) >= THRESHOLD)
    ), None)
    report = {
        "status": "smoke test, not research results" if smoke else "completed",
        "layer_numbering": "1-based decoder block outputs",
        "layers_tested": layers_tested,
        "mean_probability_change_by_condition": {
            "same_final": float(np.mean([
                float(r["delta_p_donor_rule"]) for r in rows
                if r["condition"] == "same" and r["position"] == "final"
            ])),
            "different_final": float(np.mean([
                float(r["delta_p_donor_rule"]) for r in rows
                if r["condition"] == "different" and r["position"] == "final"
            ])),
        },
        "highest_mean_different_rule_effect": {
            "layer": peak, "delta_probability": final_different[peak]
        },
        "largest_increase_from_previous_layer": (
            {
                "layer": biggest_step,
                "increase": final_different[biggest_step] - final_different[biggest_step - 1]
            } if biggest_step is not None else None
        ),
        "first_layer_above_control_by_0_05": first_over_control,
        "control_cutoff": THRESHOLD,
        "cutoff_is_statistical_significance": False,
        "caution": (
            "The cutoff is a chosen effect-size check, not a significance test. "
            "A patching effect shows causal influence on the output, not proof of "
            "a specific Boolean computation. Answer-position bias still matters."
        ),
    }
    if smoke:
        report["largest_increase_from_previous_layer"] = None
        report["first_layer_above_control_by_0_05"] = None
    write_json(out / "summary.json", report)
    print("Saved layer effects, graph and summary in", out)
    print(json.dumps({
        "highest_effect_layer": report["highest_mean_different_rule_effect"],
        "largest_step": report["largest_increase_from_previous_layer"],
        "first_above_controls": report["first_layer_above_control_by_0_05"],
    }, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="data-1000 directory with metadata.json and images/")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help="Existing Google Drive results folder")
    parser.add_argument("--precision", choices=("4bit", "fp16", "bf16", "fp32"),
                        default="4bit")
    parser.add_argument("--pairs-per-rule", type=int, default=2, choices=range(1, 6),
                        metavar="{1,2,3,4,5}",
                        help="Number of receivers per rule (3 pairs each)")
    parser.add_argument("--rotations", type=int, choices=(1, 3), default=1,
                        help="1 = fixed X/Y/Z order; 3 = all cyclic answer label orders")
    parser.add_argument("--previous-token-control", action="store_true",
                        help="Also patch the token just before the final one")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true",
                      help="Check sets and pairings without loading the model")
    mode.add_argument("--smoke-test", action="store_true",
                      help="Run one same-rule pair and one different-rule pair at the first and last layers")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    base = args.experiment_dir.expanduser().resolve()
    if not args.validate_only:
        base = require_results_root(base)
    exp.configure_data(data_dir, base / "activation_patching" / "qwen3_5")
    if len(exp.records) != 1000:
        raise ValueError("Experiment 03 requires the 1,000-image dataset.")
    sets = make_sets(exp)
    pairs = make_pairs(sets, args.pairs_per_rule)
    report = check_sets(sets, pairs, exp.records)
    print("Data and pairings:", json.dumps(report, indent=2))
    if args.validate_only:
        print("Validation passed. No GPU or model inference used.")
        return

    if args.smoke_test:
        pairs = [
            next(p for p in pairs if p["condition"] == "same"),
            next(p for p in pairs if p["condition"] == "different"),
        ]

    run_name = (
        f'pairs{args.pairs_per_rule}_rot{args.rotations}_prev{int(args.previous_token_control)}'
        + ("_smoke" if args.smoke_test else "")
    )
    out = exp.OUT / run_name
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = data_fingerprint(data_dir, sets, pairs)
    config = {
        "version": 1, "model": "Qwen/Qwen3.5-4B", "dataset": "dataset-v1",
        "dataset_fingerprint": fingerprint, "precision": args.precision,
        "seed": 2026, "pairs_per_rule": args.pairs_per_rule,
        "rotations": args.rotations, "previous_token_control": args.previous_token_control,
        "smoke_test": args.smoke_test, "pairs": pairs,
        "test_split": split_report(sets), "probabilities": "softmax restricted to X/Y/Z",
        "patching": "decoder output at final or previous token; one layer at a time",
        "layer_numbers": "1-based",
        "threshold_probability_points": THRESHOLD,
    }
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Saved run has different settings. Use a separate results folder.")
    if not config_path.exists():
        write_json(config_path, config)

    exp.load_native_model(exp.MODEL_ID, args.precision)
    adapter = exp.vlm
    layers = decoder_layers(adapter.model)
    layer_count = len(layers)
    if not layer_count:
        raise RuntimeError("No decoder layers found.")
    layer_numbers = [1, layer_count] if args.smoke_test else list(range(1, layer_count + 1))
    positions = ["final", "previous"] if args.previous_token_control else ["final"]
    print(f"Found {layer_count} decoder layers. Testing: {layer_numbers}")
    jobs = len(pairs) * args.rotations * len(layer_numbers) * len(positions)
    print(f"Expected patching forwards: {jobs} (plus original set forwards).")

    by_id = {s["set_id"]: s for s in sets}
    used = sorted({
        pair[key] for pair in pairs for key in ("receiver_set", "donor_set")
    })
    results_path = out / "patches.csv"
    done = saved_rows(results_path)
    if len(done) > jobs:
        raise ValueError("More saved rows than the current run plan.")
    if len(done) == jobs:
        print("All patching trials were already saved.")
        summarize(results_path, out, layer_count, args.smoke_test,
                  args.previous_token_control)
        return

    rotations = exp.layouts("cyclic")[:args.rotations]
    with results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()

        for rotation_index, (mapping, display) in enumerate(rotations):
            del display
            print(f"Option rotation {rotation_index + 1}/{len(rotations)}")
            baseline = {}
            for set_id in used:
                item = by_id[set_id]
                inputs = get_inputs(adapter, item, mapping)
                baseline[set_id] = record_original(adapter, inputs, mapping, layers)
                del inputs
                print(f"  Original: {set_id} -> {baseline[set_id]['prediction']}", flush=True)

            for pair in pairs:
                receiver = baseline[pair["receiver_set"]]
                donor = baseline[pair["donor_set"]]
                inputs = None
                for layer in layer_numbers:
                    for position in positions:
                        key = (pair["pair_id"], rotation_index, layer, position)
                        if key in done:
                            continue
                        if inputs is None:
                            inputs = get_inputs(adapter, by_id[pair["receiver_set"]], mapping)
                        donor_vector = donor["activations"][layer][position]
                        prob = patched_probabilities(
                            adapter, inputs, mapping, layers[layer - 1],
                            donor_vector, position
                        )
                        row = to_row(pair, rotation_index, layer, position,
                                     receiver, donor, prob)
                        writer.writerow(row)
                        handle.flush()
                        done[key] = row
                        print(
                            f'{len(done)}/{jobs} {pair["condition"]} '
                            f'{pair["receiver_rule"]}->{pair["donor_rule"]} '
                            f'layer={layer} {position} '
                            f'delta_P={row["delta_p_donor_rule"]:+.4f} '
                            f'prediction={row["patched_prediction"]}',
                            flush=True,
                        )
                del inputs
            del baseline

    if len(done) != jobs:
        raise RuntimeError(f"Only {len(done)} of {jobs} patching results saved.")
    summarize(results_path, out, layer_count, args.smoke_test,
              args.previous_token_control)


if __name__ == "__main__":
    main()
