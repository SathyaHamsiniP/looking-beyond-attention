"""Manifest patching with fixed receiver-target metrics and matched pair plans."""

import argparse
from pathlib import Path

from .data_loader import load_manifest, option_labels, prompt
from .model_qwen35 import decoder_layers, load, model_metadata, score
from .run_utils import (
    Journal,
    atomic_json,
    dataset_fingerprint,
    digest,
    file_fingerprint,
    prepare_run,
)


def make_pairs(sets, partition, pairs_per_rule=1):
    by_rule = {
        r: sorted(
            [s for s in sets if s["rule"] == r and s["partition"] == partition],
            key=lambda x: x["set_id"],
        )
        for r in sorted({s["rule"] for s in sets})
    }
    pairs = []
    for rule, items in by_rule.items():
        if len(items) < pairs_per_rule:
            raise ValueError("Insufficient receivers")
        accepted = 0
        for receiver in items:
            names = {o["filename"] for o in receiver["observations"]}
            selected = {
                other: next(
                    (
                        s
                        for s in donors
                        if s["set_id"] != receiver["set_id"]
                        and names.isdisjoint(o["filename"] for o in s["observations"])
                    ),
                    None,
                )
                for other, donors in by_rule.items()
            }
            if any(s is None for s in selected.values()):
                continue
            for other, donor in selected.items():
                pairs.append(
                    dict(
                        pair_id=receiver["set_id"] + "__" + donor["set_id"],
                        receiver=receiver["set_id"],
                        donor=donor["set_id"],
                        receiver_rule=rule,
                        donor_rule=other,
                        condition="same" if rule == other else "different",
                    )
                )
            accepted += 1
            if accepted == pairs_per_rule:
                break
        if accepted < pairs_per_rule:
            raise ValueError(
                f"Insufficient image-disjoint donors for {rule}; generate more sets per rule"
            )
    return pairs


def capture(adapter, inputs, mapping, blocks, position):
    index = inputs["input_ids"].shape[1] - (1 if position == "final" else 2)
    if index < 0:
        raise ValueError("Insufficient prompt tokens")
    vectors = {}
    handles = []

    def hook(layer):
        def collect(module, args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            vectors[layer] = hidden[0, index].detach().cpu().clone()

        return collect

    try:
        for n, block in enumerate(blocks, 1):
            handles.append(block.register_forward_hook(hook(n)))
        probs = score(adapter, inputs, mapping)
    finally:
        for handle in handles:
            handle.remove()
    if len(vectors) != len(blocks):
        raise ValueError("Missing decoder captures")
    return probs, vectors


def patched(adapter, inputs, mapping, block, vector, position):
    index = inputs["input_ids"].shape[1] - (1 if position == "final" else 2)

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if vector.shape != hidden[0, index].shape:
            raise ValueError("Hidden size mismatch")
        changed = hidden.clone()
        changed[0, index] = vector.to(changed)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        return changed

    handle = block.register_forward_hook(hook)
    try:
        return score(adapter, inputs, mapping)
    finally:
        handle.remove()


def main(experiment=3):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-dir", "results-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--partition", choices=("validation", "test"), default="validation")
    p.add_argument("--pairs-per-rule", type=int, default=1)
    p.add_argument("--rotations", type=int, default=0)
    p.add_argument("--previous-token-control", action="store_true")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--precision", default="4bit")
    p.add_argument("--revision", default="main")
    p.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    m = load_manifest(a.manifest, a.data_dir)
    if not 0 <= a.rotations <= len(m["rules"]) or a.pairs_per_rule < 1:
        p.error("Invalid rotation or pair count")
    pairs = make_pairs(m["sets"], a.partition, a.pairs_per_rule)
    if a.smoke_test:
        pairs = [
            next(x for x in pairs if x["condition"] == c) for c in ("same", "different")
        ]
    plan = dict(
        pairs=pairs,
        partition=a.partition,
        rotations=a.rotations or len(m["rules"]),
        positions=["final", "previous"] if a.previous_token_control else ["final"],
        rules=list(m["rules"]),
        manifest_hash=digest(m),
        layers=a.layers,
        smoke_test=a.smoke_test,
        max_pixels=a.max_pixels,
        precision=a.precision,
        revision=a.revision,
    )
    if a.validate_only:
        print(f"Validated {len(pairs)} image-disjoint pairs. No inference.")
        return
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
    blocks = decoder_layers(adapter.model)
    layers = a.layers or (
        [1, len(blocks)] if a.smoke_test else list(range(1, len(blocks) + 1))
    )
    if len(set(layers)) != len(layers) or any(x < 1 or x > len(blocks) for x in layers):
        raise ValueError("Invalid layers")
    plan["layers"] = layers
    config = dict(
        plan=plan,
        plan_hash=digest(plan),
        adapter=str(a.adapter) if a.adapter else None,
        data_fingerprint=dataset_fingerprint(m, a.data_dir),
        model=model_metadata(adapter, option_labels(len(m["rules"]))),
    )
    if a.adapter:
        config["adapter_hashes"] = {
            f.name: file_fingerprint(f)
            for f in sorted(a.adapter.glob("*"))
            if f.is_file()
        }
    out = prepare_run(a.results_root, experiment, m, config)
    atomic_json(out / "pair_plan.json", plan)
    journal = Journal(out / "patches.jsonl")
    items = {s["set_id"]: s for s in m["sets"]}
    for rotation in range(plan["rotations"]):
        order = list(m["rules"])
        order = order[rotation:] + order[:rotation]
        for pair in pairs:
            receiver = {**items[pair["receiver"]], "option_order": order}
            donor = {**items[pair["donor"]], "option_order": order}
            paths, text, mapping = prompt(receiver, m, a.data_dir)
            inputs = adapter._inputs(paths, text)
            for position in plan["positions"]:
                keys = [
                    f"{pair['pair_id']}:{rotation}:{position}:{layer}"
                    for layer in layers
                ]
                if all(k in journal.rows for k in keys):
                    continue
                before, _ = capture(adapter, inputs, mapping, blocks, position)
                paths, text, _ = prompt(donor, m, a.data_dir)
                donor_inputs = adapter._inputs(paths, text)
                donor_probs, vectors = capture(
                    adapter, donor_inputs, mapping, blocks, position
                )
                del donor_inputs
                for layer, key in zip(layers, keys):
                    if key in journal.rows:
                        continue
                    after = patched(
                        adapter,
                        inputs,
                        mapping,
                        blocks[layer - 1],
                        vectors[layer],
                        position,
                    )
                    pred = max(after, key=after.get)
                    original = max(before, key=before.get)
                    donor_pred = max(donor_probs, key=donor_probs.get)
                    journal.add(
                        key,
                        dict(
                            **pair,
                            rotation=rotation,
                            position=position,
                            layer=layer,
                            receiver_prediction=original,
                            donor_prediction=donor_pred,
                            patched_prediction=pred,
                            receiver_correct=original == pair["receiver_rule"],
                            donor_correct=donor_pred == pair["donor_rule"],
                            receiver_probabilities=before,
                            donor_probabilities=donor_probs,
                            patched_probabilities=after,
                            delta_receiver_probability=after[pair["receiver_rule"]]
                            - before[pair["receiver_rule"]],
                            delta_donor_probability=after[pair["donor_rule"]]
                            - before[pair["donor_rule"]],
                            changed_answer=pred != original,
                        ),
                    )
                del vectors
            del inputs
    from .comparison import summarize

    summarize(list(journal.rows.values()), out)
