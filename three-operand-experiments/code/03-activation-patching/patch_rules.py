"""Final-token decoder-output patching pilot for Qwen3.5-4B."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / '01-operator-controls'))
sys.path.insert(0, str(CODE / '02-probing'))
import operator_controls_qwen35 as experiment
from probe_concepts import decoder_layers
from probe_rules import write_csv
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root


def make_pairs(records, count=1):
    if not 1 <= count <= 6:
        raise ValueError('Use 1 to 6 pairs per rule pair; this dataset has only six variations.')
    pairs = []
    for pair_number, (first, second) in enumerate(combinations(experiment.RULES, 2)):
        shared = [r for r in records if experiment.satisfies(first, r['bits']) and experiment.satisfies(second, r['bits'])]
        for index in range(count):
            rng = random.Random(2026 + 100 * experiment.RULES.index(first) + 10 * experiment.RULES.index(second) + index)
            # Cover all common types before sampling the remaining shared images.
            types = sorted({tuple(r['bits']) for r in shared})
            common = [rng.choice([r for r in shared if tuple(r['bits']) == bits]) for bits in types]
            chosen = {r['filename'] for r in common}
            common += rng.sample([r for r in shared if r['filename'] not in chosen], 9 - len(common))
            slot = (3 * pair_number + index) % 10
            scenes = {}
            for rule in (first, second):
                diagnostic = next(r for r in records if sum(r['bits']) == 1
                                  and experiment.satisfies(rule, r['bits']) and r['variation'] == index)
                control = next(r for r in records if r['bits'] == diagnostic['bits'] and r['variation'] == (index + 1) % 6)
                for suffix, record in [('', diagnostic), ('_same', control)]:
                    items = common.copy()
                    items.insert(slot, record)
                    scenes[rule + suffix] = dict(image_files=[r['filename'] for r in items],
                                                scene_bits=[r['bits'] for r in items])
                    assert len(set(scenes[rule + suffix]['image_files'])) == 10
                    assert experiment.compatible_rules(items) == [rule]
            left, right = scenes[first]['image_files'], scenes[second]['image_files']
            assert sum(a == b for a, b in zip(left, right)) == 9
            assert left[slot] != right[slot]
            pairs.append(dict(pair_id=f'{first}_{second}_{index}', rules=[first, second], diagnostic_slot=slot,
                              scenes=scenes))
    return pairs


def forward(adapter, paths, prompt, token_ids, patch_layer=None, replacement=None, capture=False):
    """Capture CPU copies, or replace one block's last-token output on a fresh prefill."""
    import torch
    blocks = decoder_layers(adapter.model)
    states, handles = [], []
    if patch_layer is not None and not 0 <= patch_layer < len(blocks):
        raise ValueError('Patch layer is outside the decoder.')

    def hook_for(index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1 or not hidden.is_floating_point():
                raise ValueError('Expected one floating-point decoder sequence.')
            if capture:
                states.append(hidden[0, -1].detach().to('cpu').clone())
            if index == patch_layer:
                if replacement is None or replacement.ndim != 1 or replacement.shape[0] != hidden.shape[-1]:
                    raise ValueError('Donor and recipient hidden widths must match.')
                if not replacement.is_floating_point() or not torch.isfinite(replacement).all():
                    raise ValueError('Donor state must be finite floating point.')
                donor = replacement.to(device=hidden.device, dtype=hidden.dtype)
                if not torch.isfinite(donor).all():
                    raise ValueError('Donor conversion produced non-finite values.')
                # Never change the model output or saved donor tensor in place.
                patched = hidden.clone()
                patched[0, -1, :] = donor
                return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return hook

    try:
        for index, block in enumerate(blocks):
            if capture or index == patch_layer:
                handles.append(block.register_forward_hook(hook_for(index)))
        with torch.inference_mode():
            inputs = adapter._inputs(paths, prompt)
            output = adapter.model(**inputs, use_cache=False, logits_to_keep=1)
            logits = output.logits[0, -1, [token_ids[label] for label in experiment.LABELS]].float().cpu()
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite answer logits.')
    finally:
        for handle in handles:
            handle.remove()
    if capture and len(states) != len(blocks):
        raise ValueError('Not every decoder block executed once.')
    return dict(logits=logits, states=states)


def rule_values(logits, mapping):
    values = np.asarray(logits, dtype=float)
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return ({mapping[label]: float(values[i]) for i, label in enumerate(experiment.LABELS)},
            {mapping[label]: float(probabilities[i]) for i, label in enumerate(experiment.LABELS)})


def effect_row(pair, layout, mapping, recipient, donor, control, layer, base, donor_base, patched, tolerance):
    b, bp = rule_values(base, mapping)
    d, dp = rule_values(donor_base, mapping)
    p, pp = rule_values(patched, mapping)
    error = float(np.max(np.abs(np.asarray(patched) - np.asarray(base))))
    donor_error = float(np.max(np.abs(np.asarray(patched) - np.asarray(donor_base))))
    baseline_gap = d[donor] - d[recipient] - (b[donor] - b[recipient])
    source = donor if control == 'cross' else recipient + ('_same' if control == 'same_rule' else '')
    row = dict(pair_id=pair['pair_id'], layout=layout, recipient_rule=recipient, donor_rule=donor,
               patch_source_scene=source, patch_source_rule=donor if control == 'cross' else recipient,
               control=control, layer=layer + 1, recipient_prediction=max(bp, key=bp.get),
               donor_prediction=max(dp, key=dp.get), same_baseline_prediction=max(bp, key=bp.get) == max(dp, key=dp.get),
               baseline_rule_logit_gap=baseline_gap,
               nearly_identical_baselines=max(abs(d[r] - b[r]) for r in experiment.RULES) <= tolerance,
               baseline_probability_difference=max(abs(dp[r] - bp[r]) for r in experiment.RULES),
               nearly_identical_rule_margins=abs(baseline_gap) <= tolerance,
               recipient_margin=b[donor] - b[recipient], patched_margin=p[donor] - p[recipient],
               margin_change=(p[donor] - p[recipient]) - (b[donor] - b[recipient]),
               max_error_from_recipient=error, max_error_from_donor=donor_error,
               self_check_pass=(error <= tolerance) if control == 'self' else '',
               final_transfer_pass=(donor_error <= tolerance) if control == 'cross' and layer == pair['last_layer'] else '')
    for rule in experiment.RULES:
        row.update({f'recipient_logit_{rule}': b[rule], f'donor_logit_{rule}': d[rule],
                    f'patched_logit_{rule}': p[rule], f'recipient_probability_{rule}': bp[rule],
                    f'donor_probability_{rule}': dp[rule], f'patched_probability_{rule}': pp[rule]})
    return row


def save_plot(rows, out):
    import matplotlib.pyplot as plt
    for control in ('cross', 'same_rule', 'self'):
        layers = sorted({r['layer'] for r in rows if r['control'] == control})
        means = [np.mean([r['margin_change'] for r in rows if r['control'] == control and r['layer'] == layer]) for layer in layers]
        plt.plot(layers, means, marker='.', label=control)
    plt.axhline(0, color='grey', linestyle=':')
    plt.xlabel('Patched decoder block output (1-based)')
    plt.ylabel('Change in donor-rule minus recipient-rule logit')
    plt.title('Final-token activation patching pilot (reused images)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'patching_effects.png', dpi=160)
    plt.close()


def baseline_row(input_id, result):
    logits = result['logits'].numpy()
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    rule_logits, rule_probability = rule_values(logits, result['mapping'])
    row = dict(input_id=input_id, pair_id=result['pair_id'], layout=result['layout'], scene=result['scene_id'])
    row.update({f'logit_{label}': float(logits[i]) for i, label in enumerate(experiment.LABELS)})
    row.update({f'probability_{label}': float(probability[i]) for i, label in enumerate(experiment.LABELS)})
    row.update({f'logit_{rule}': rule_logits[rule] for rule in experiment.RULES})
    row.update({f'probability_{rule}': rule_probability[rule] for rule in experiment.RULES})
    return row


def save_checks(rows, out, tolerance):
    checks = dict(failed_rows=sum(r['self_check_pass'] is False or r['final_transfer_pass'] is False for r in rows),
                  self_checks=sum(r['self_check_pass'] != '' for r in rows),
                  final_checks=sum(r['final_transfer_pass'] != '' for r in rows), tolerance=tolerance)
    (out / 'numerical_checks.json').write_text(json.dumps(checks, indent=2))
    return checks['failed_rows']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=CODE.parent / 'data')
    parser.add_argument('--experiment-dir', type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--pairs-per-rule-pair', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=0.05, help='Absolute restricted-logit tolerance for numerical checks')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error('--tolerance must be positive')
    out = args.experiment_dir.expanduser().resolve()
    if not args.validate_only:
        out = require_results_root(out)
    out = out / 'activation-patching/qwen3_5'
    if args.smoke_test:
        out = out / 'smoke'
    experiment.configure_data(args.data_dir, out)
    pairs = make_pairs(experiment.records, args.pairs_per_rule_pair)
    images = {name for pair in pairs for scene in pair['scenes'].values() for name in scene['image_files']}
    print(f'Validated {len(pairs)} matched pairs, {len(images)} unique images; nine common slots per pair.')
    print('Pairs reuse images; rotations and directions are repeated measurements, not independent samples.')
    if args.validate_only:
        return
    if args.smoke_test:
        pairs = pairs[:1]
    digest = hashlib.sha256()
    for record in sorted(experiment.records, key=lambda r: r['filename']):
        digest.update(record['filename'].encode())
        digest.update((experiment.IMAGE_DIR / record['filename']).read_bytes())
    versions = {}
    for package in ('torch', 'transformers', 'accelerate', 'bitsandbytes'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    layouts = experiment.layouts('cyclic')[:1] if args.smoke_test else experiment.layouts('cyclic')
    config = dict(version=1, model=experiment.MODEL_ID, precision='4bit', seed=2026, pairs=pairs,
                  metadata_hash=experiment.DATA_HASH, image_hash=digest.hexdigest(), software=versions,
                  smoke_test=args.smoke_test, layouts=[m for m, display in layouts], tolerance=args.tolerance,
                  prompt=experiment.build_input(pairs[0]['scenes'][pairs[0]['rules'][0]], layouts[0][0], layouts[0][1])[1])
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Existing configuration differs; use another existing results root rather than mix runs.')
    temporary = out / 'config.tmp.json'
    temporary.write_text(json.dumps(config, indent=2))
    temporary.replace(path)
    checkpoint = out / 'patches.jsonl'
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()] if checkpoint.exists() else []
    if any(r['self_check_pass'] is False or r['final_transfer_pass'] is False for r in rows):
        raise SystemExit('A saved numerical control failed. Inspect it before starting a fresh run; it will not be skipped.')
    import torch
    torch.manual_seed(2026)
    experiment.load_native_model(experiment.MODEL_ID, '4bit')
    adapter = experiment.vlm
    count = len(decoder_layers(adapter.model))
    layers = sorted({0, count // 2, count - 1}) if args.smoke_test else list(range(count))
    key = lambda row: (row['pair_id'], row['layout'], row['recipient_rule'], row['control'], row['layer'])
    done = {key(row) for row in rows}
    expected = {(pair['pair_id'], layout, recipient, control, layer + 1)
                for pair in pairs for layout in range(len(layouts)) for recipient in pair['rules']
                for control in ('cross', 'self', 'same_rule') for layer in layers}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Duplicate or unexpected patch rows in checkpoint.')
    baseline_rows = []
    for pair in pairs:
        pair['last_layer'] = count - 1
        for layout, (mapping, display) in enumerate(layouts):
            baselines = {}
            for name, scene in pair['scenes'].items():
                cache = out / f'{pair["pair_id"]}_{layout}_{name}.pt'
                if cache.exists():
                    result = torch.load(cache, map_location='cpu', weights_only=True)
                else:
                    paths, prompt = experiment.build_input(scene, mapping, display)
                    result = forward(adapter, paths, prompt, experiment.TOKEN_IDS, capture=True)
                    result.update(mapping=mapping, scene_id=name, pair_id=pair['pair_id'], layout=layout)
                    temporary = cache.with_suffix('.tmp.pt')
                    torch.save(result, temporary)
                    temporary.replace(cache)
                if (len(result['states']) != count or result['logits'].shape != (3,)
                        or not torch.isfinite(result['logits']).all()
                        or result['mapping'] != mapping or result['scene_id'] != name
                        or result['pair_id'] != pair['pair_id'] or result['layout'] != layout
                        or any(h.ndim != 1 or not h.is_floating_point() or not torch.isfinite(h).all()
                               for h in result['states'])):
                    raise ValueError('Baseline checkpoint metadata, shape or values differ.')
                baselines[name] = result
                baseline_rows.append(baseline_row(cache.stem, result))
            write_csv(out / 'baselines.csv', baseline_rows)
            first, second = pair['rules']
            for recipient, donor in [(first, second), (second, first)]:
                paths, prompt = experiment.build_input(pair['scenes'][recipient], mapping, display)
                baseline = baselines[recipient]['logits'].numpy()
                donor_logits = baselines[donor]['logits'].numpy()
                if np.argmax(baseline) == np.argmax(donor_logits):
                    print(f'{pair["pair_id"]} layout {layout}: same baseline answer; patching is not error recovery.')
                if np.max(np.abs(baseline - donor_logits)) <= args.tolerance:
                    print(f'{pair["pair_id"]} layout {layout}: donor and recipient logits nearly identical; no recovery claim.')
                for layer in layers:
                    for control, source in [('cross', donor), ('self', recipient), ('same_rule', recipient + '_same')]:
                        identity = (pair['pair_id'], layout, recipient, control, layer + 1)
                        if identity in done:
                            continue
                        patched = forward(adapter, paths, prompt, experiment.TOKEN_IDS, layer,
                                          baselines[source]['states'][layer])['logits'].numpy()
                        row = effect_row(pair, layout, mapping, recipient, donor, control, layer,
                                         baseline, donor_logits, patched, args.tolerance)
                        rows.append(row)
                        temporary = out / 'patches.tmp.jsonl'
                        temporary.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                        temporary.replace(checkpoint)
                        done.add(identity)
                        if row['self_check_pass'] is False or row['final_transfer_pass'] is False:
                            write_csv(out / 'patches.csv', rows)
                            save_checks(rows, out, args.tolerance)
                            raise SystemExit(f'Numerical check FAILED: {identity}. Diagnostics saved; scan stopped.')
                    if layer % 8 == 0 or layer == layers[-1]:
                        print(f'{pair["pair_id"]} layout {layout + 1}: {donor} -> {recipient}, layer {layer + 1}/{count}')
            print(f'Finished {pair["pair_id"]}, label layout {layout + 1}/{len(layouts)}')
    if done != expected:
        raise ValueError('Layer scan did not complete all planned patches.')
    write_csv(out / 'patches.csv', rows)
    save_plot(rows, out)
    save_checks(rows, out, args.tolerance)
    print('Outputs saved. Final-layer transfer is an implementation check, not evidence of Boolean reasoning.')


if __name__ == '__main__':
    main()
