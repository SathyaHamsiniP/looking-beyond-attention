"""Run a new matched patching baseline or adapter run. Keep Experiment 03 untouched."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04-lora"))
from common import *
from build_training_sets import load, RULES


def make_pairs(manifest, partition, receivers_per_rule):
    if not 1 <= receivers_per_rule <= 10:
        raise ValueError("Use 1..10 receivers per rule")
    groups = {
        r: [
            s
            for s in manifest["sets"]
            if s["partition"] == partition and s["rule"] == r
        ]
        for r in RULES
    }
    pairs = []
    for rule in RULES:
        for receiver in groups[rule][:receivers_per_rule]:
            files = set(receiver["image_files"])
            for donor_rule in RULES:
                candidates = [
                    s for s in groups[donor_rule] if s["set_id"] != receiver["set_id"]
                ]
                donor = min(
                    candidates,
                    key=lambda s: (
                        len(files.intersection(s["image_files"])),
                        s["set_id"],
                    ),
                )
                pairs.append(
                    dict(
                        pair_id=receiver["set_id"] + "__" + donor["set_id"],
                        receiver=receiver["set_id"],
                        donor=donor["set_id"],
                        receiver_rule=rule,
                        donor_rule=donor_rule,
                        condition="same" if rule == donor_rule else "different",
                        shared_images=sorted(files.intersection(donor["image_files"])),
                    )
                )
    return pairs


def capture(adapter, prepared, rotation, blocks):
    vectors = {}
    handles = []
    index = prepared["input_ids"].shape[1]

    def hook(number):
        def collect(module, args, result):
            hidden = result[0] if isinstance(result, (tuple, list)) else result
            # Fixed prompt indices also work while scoring multi-token answers.
            vectors[number] = {
                "final": hidden[0, index - 1].detach().cpu().clone(),
                "previous": hidden[0, index - 2].detach().cpu().clone(),
            }

        return collect

    try:
        for number, block in enumerate(blocks, 1):
            handles.append(block.register_forward_hook(hook(number)))
        probabilities = score(adapter, prepared, rotation)
    finally:
        for handle in handles:
            handle.remove()
    if len(vectors) != len(blocks):
        raise ValueError("Missing decoder vectors")
    return probabilities, vectors


def patch(adapter, prepared, rotation, block, vector, position):
    index = prepared["input_ids"].shape[1] - (1 if position == "final" else 2)

    def replace(module, args, result):
        hidden = result[0] if isinstance(result, (tuple, list)) else result
        changed = hidden.clone()
        if vector.shape != changed[0, index].shape:
            raise ValueError("Donor hidden size differs")
        changed[0, index] = vector.to(changed)
        if isinstance(result, tuple):
            return (changed, *result[1:])
        if isinstance(result, list):
            return [changed, *result[1:]]
        return changed

    handle = block.register_forward_hook(replace)
    try:
        return score(adapter, prepared, rotation)
    finally:
        handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_args(p)
    p.add_argument("--partition", choices=("validation", "test"), default="validation")
    p.add_argument("--receivers-per-rule", type=int, default=1)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 24, 32])
    p.add_argument(
        "--positions",
        nargs="+",
        choices=("final", "previous"),
        default=["final", "previous"],
    )
    a = p.parse_args()
    m = load(a.manifest, a.data_dir)
    config = settings(a)
    if (
        len(set(a.layers)) != len(a.layers)
        or not all(1 <= n <= 32 for n in a.layers)
        or len(set(a.positions)) != len(a.positions)
    ):
        p.error("Invalid layers or positions")
    pairs = make_pairs(m, a.partition, a.receivers_per_rule)
    if a.smoke_test:
        pairs = [
            next(p for p in pairs if p["condition"] == c) for c in ("same", "different")
        ]
        a.layers = [1, 32]
    plan = dict(
        pairs=pairs,
        layers=a.layers,
        positions=a.positions,
        rotations=[0, 1, 2],
        mappings=[mapping(r) for r in range(3)],
        partition=a.partition,
    )
    if a.validate_only:
        print(
            "Checked pairs:",
            len(pairs),
            "trials:",
            len(pairs) * 3 * len(a.layers) * len(a.positions),
            "overlapping pairs:",
            sum(bool(p["shared_images"]) for p in pairs),
        )
        return
    if a.adapter:
        meta = json.loads((a.adapter / "experiment.json").read_text())
        if meta["manifest_hash"] != digest(m):
            raise ValueError("Adapter manifest differs")
    out = run_folder(
        a.output_root,
        dict(
            kind="patching",
            settings=config,
            plan=plan,
            smoke_test=a.smoke_test,
            adapter=adapter_identity(a.adapter),
        ),
        m,
    )
    save(out / "pair_plan.json", plan)
    journal = out / "patches.jsonl"
    done = rows(journal)
    expected = {
        f"{p['pair_id']}:{r}:{layer}:{position}"
        for p in pairs
        for r in range(3)
        for layer in a.layers
        for position in a.positions
    }
    if not set(done) <= expected:
        raise ValueError("Unexpected saved trial keys")
    adapter = load_model(a)
    blocks = layers(adapter.model)
    if len(blocks) != 32:
        raise ValueError(
            "Expected 32 Qwen3.5-4B decoder layers; inspect model revision"
        )
    by_id = {s["set_id"]: s for s in m["sets"]}
    for rotation in range(3):
        for pair in pairs:
            if all(
                f"{pair['pair_id']}:{rotation}:{l}:{p}" in done
                for l in a.layers
                for p in a.positions
            ):
                continue
            receiver_input = inputs(
                adapter, by_id[pair["receiver"]], a.data_dir, rotation
            )
            before, _ = capture(adapter, receiver_input, rotation, blocks)
            donor_input = inputs(adapter, by_id[pair["donor"]], a.data_dir, rotation)
            donor, vectors = capture(adapter, donor_input, rotation, blocks)
            del donor_input
            receiver_prediction = max(before, key=before.get)
            donor_prediction = max(donor, key=donor.get)
            for layer in a.layers:
                for position in a.positions:
                    key = f"{pair['pair_id']}:{rotation}:{layer}:{position}"
                    if key in done:
                        continue
                    after = patch(
                        adapter,
                        receiver_input,
                        rotation,
                        blocks[layer - 1],
                        vectors[layer][position],
                        position,
                    )
                    prediction = max(after, key=after.get)
                    row = dict(
                        **pair,
                        rotation=rotation,
                        mapping=mapping(rotation),
                        layer=layer,
                        position=position,
                        receiver_probabilities=before,
                        donor_probabilities=donor,
                        patched_probabilities=after,
                        receiver_prediction=receiver_prediction,
                        donor_prediction=donor_prediction,
                        patched_prediction=prediction,
                        receiver_correct=receiver_prediction == pair["receiver_rule"],
                        donor_correct=donor_prediction == pair["donor_rule"],
                        delta_donor=after[pair["donor_rule"]]
                        - before[pair["donor_rule"]],
                        delta_receiver=after[pair["receiver_rule"]]
                        - before[pair["receiver_rule"]],
                        changed_answer=prediction != receiver_prediction,
                        changed_to_donor_true=prediction != receiver_prediction
                        and prediction == pair["donor_rule"],
                        changed_to_donor_prediction=prediction != receiver_prediction
                        and prediction == donor_prediction,
                    )
                    append(journal, key, row)
                    done[key] = row
            del vectors, receiver_input
            memory()
    if set(done) != expected:
        raise ValueError("Incomplete patching results")
    save(out / "complete.json", dict(trials=len(done)))


if __name__ == "__main__":
    main()
