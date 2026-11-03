"""Image-disjoint rule-probing pilot using the existing 48-image dataset."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
from probe_concepts import POOLS, capture
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root

SEED = 2026
PROMPT = '''These ten images all satisfy one hidden Boolean rule.
A is true when a green cube is present.
B is true when a blue sphere is present.
C is true when a red cylinder is present.
Each concept requires the exact colour and shape. Other objects are distractors.
Consider all ten images and identify the Boolean relationship that explains them.'''


def make_sets(experiment):
    if len(experiment.records) == 1000:
        # Make separate ten-image sets. A picture is never reused.
        # The rule is clear because all five positive ABC patterns appear.
        sets = []
        for split, count in (('train', 8), ('test', 5)):
            rng = random.Random(SEED + (0 if split == 'train' else 100000))
            by_bits = {}
            for bits in product((0, 1), repeat=3):
                by_bits[bits] = [
                    r for r in experiment.records
                    if r['split'] == split and tuple(r['bits']) == bits
                ]
                rng.shuffle(by_bits[bits])
            for rule_index, rule in enumerate(experiment.RULES):
                positive = [
                    bits for bits in product((0, 1), repeat=3)
                    if experiment.satisfies(rule, bits)
                ]
                unique = next(bits for bits in positive if sum(bits) == 1)
                shared = [bits for bits in positive if bits != unique]
                for index in range(count):
                    set_index = rule_index * count + index
                    # 5 patterns, then 3 extra images of the unique type and
                    # 2 extra shared types. This fits in the 200-image test split.
                    needed = positive + [unique] * 3 + [
                        shared[(2 * set_index + offset) % len(shared)]
                        for offset in range(2)
                    ]
                    if any(len(by_bits[bits]) < needed.count(bits) for bits in set(needed)):
                        raise ValueError(f'Not enough unused {split} images for {rule}.')
                    chosen = [by_bits[bits].pop() for bits in needed]
                    rng.shuffle(chosen)
                    assert experiment.compatible_rules(chosen) == [rule]
                    sets.append(dict(
                        set_id=f'{split}_{rule}_{index:02d}',
                        split=split, rule=rule,
                        image_files=[r['filename'] for r in chosen],
                        bits=[r['bits'] for r in chosen],
                    ))
        names = [name for item in sets for name in item['image_files']]
        assert len(names) == len(set(names))
        return sets

    # Keep the original 48-image pilot unchanged.
    original_pool = experiment.POOL
    sets = []
    try:
        for split, variations, count in [('train', range(4), 8), ('test', range(4, 6), 1)]:
            experiment.POOL = [r for r in experiment.records if r['variation'] in variations]
            seen = set()
            for rule in experiment.RULES:
                for index in range(count):
                    scene = experiment.sample_set(rule, index)
                    identity = tuple(sorted(scene['image_files']))
                    if identity in seen:
                        raise ValueError('Duplicate underlying scene set; do not count reordered sets twice.')
                    seen.add(identity)
                    assert len(identity) == len(set(identity)) == 10
                    assert experiment.compatible_rules(scene['items']) == [rule]
                    sets.append(dict(set_id=f'{split}_{rule}_{index:02d}', split=split,
                                     rule=rule, image_files=scene['image_files'], bits=scene['scene_bits']))
    finally:
        experiment.POOL = original_pool
    train_images = {name for s in sets if s['split'] == 'train' for name in s['image_files']}
    test_images = {name for s in sets if s['split'] == 'test' for name in s['image_files']}
    assert train_images.isdisjoint(test_images)
    return sets

def split_report(sets):
    if len(sets) == 39:
        used = [name for item in sets for name in item['image_files']]
        assert len(used) == len(set(used))
        return {
            'evaluation': 'No image is reused across sets.',
            'chance_accuracy': 1 / 3,
            'dataset_images': 1000,
            'train': {'sets': 24, 'images_used': 240, 'sets_per_rule': 8},
            'test': {'sets': 15, 'images_used': 150, 'sets_per_rule': 5},
        }

    report = {'evaluation': 'image-disjoint pilot; overlapping sets within each partition',
              'chance_accuracy': 1 / 3, 'dataset_images': 48,
              'unused_images': 48 - len({n for s in sets for n in s['image_files']})}
    for split in ('train', 'test'):
        subset = [s for s in sets if s['split'] == split]
        images = [set(s['image_files']) for s in subset]
        # With 28/14 images there can be at most two/one disjoint ten-image sets.
        disjoint = max(k for k in range(1, len(set.union(*images)) // 10 + 1)
                       if any(sum(len(x) for x in group) == len(set.union(*group))
                              for group in combinations(images, k)))
        report[split] = dict(unique_sets=len(subset), presentations=len(subset),
                             max_mutually_image_disjoint_sets=disjoint,
                             unique_underlying_images=len({n for s in subset for n in s['image_files']}),
                             sets_per_rule=dict(Counter(s['rule'] for s in subset)))
    return report


def scene_counts(scene):
    counts = Counter(tuple(bits) for bits in scene['bits'])
    return [counts[bits] / 10 for bits in product((0, 1), repeat=3)]


def diagnostic_rule(scene):
    # Under coverage sampling, the one-hot positive type identifies the rule.
    diagnostic = {(0, 0, 1): 'R1', (0, 1, 0): 'R2', (1, 0, 0): 'R3'}
    found = {diagnostic[tuple(bits)] for bits in scene['bits'] if tuple(bits) in diagnostic}
    if len(found) != 1:
        raise ValueError('Expected exactly one diagnostic configuration type.')
    return found.pop()


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def score_rules(features, sets, out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    train = [i for i, s in enumerate(sets) if s['split'] == 'train']
    test = [i for i, s in enumerate(sets) if s['split'] == 'test']
    labels = np.array([s['rule'] for s in sets])
    shuffled = labels[train].copy()
    np.random.default_rng(SEED).shuffle(shuffled)
    predictions, metrics = [], []

    def record(method, layer, pool, control, predicted, probabilities, classes):
        correct = predicted == labels[test]
        metrics.append(dict(method=method, layer=layer, pool=pool, control=control,
                            accuracy=float(correct.mean()), test_sets=len(test)))
        for j, index in enumerate(test):
            row = dict(set_id=sets[index]['set_id'], method=method, layer=layer, pool=pool,
                       control=control, truth=labels[index], predicted=predicted[j], correct=int(correct[j]))
            row.update({f'probability_{rule}': float(probabilities[j, classes.index(rule)])
                        for rule in ('R1', 'R2', 'R3')})
            predictions.append(row)

    def fit(x, method, layer, pool):
        for control, target in [('measured', labels[train]), ('shuffled_train_labels', shuffled)]:
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=SEED))
            probe.fit(x[train], target)
            record(method, layer, pool, control, probe.predict(x[test]), probe.predict_proba(x[test]), list(probe.classes_))

    fit(np.array([scene_counts(s) for s in sets]), 'scene_count', 0, 'counts')
    predicted = np.array([diagnostic_rule(sets[i]) for i in test])
    classes = ['R1', 'R2', 'R3']
    probabilities = np.array([[float(r == prediction) for r in classes] for prediction in predicted])
    record('diagnostic_scene', 0, 'bits', 'measured', predicted, probabilities, classes)
    for layer in range(features.shape[1]):
        for pool, name in enumerate(POOLS):
            fit(features[:, layer, pool], 'hidden_state', layer + 1, name)
    write_csv(out / 'predictions.csv', predictions)
    write_csv(out / 'layer_accuracy.csv', metrics)
    for pool in POOLS:
        for control, style in [('measured', '-'), ('shuffled_train_labels', '--')]:
            rows = [r for r in metrics if r['method'] == 'hidden_state' and r['pool'] == pool and r['control'] == control]
            plt.plot([r['layer'] for r in rows], [r['accuracy'] for r in rows], style, label=f'{pool}: {control}')
    for method, color in [('scene_count', 'grey'), ('diagnostic_scene', 'black')]:
        accuracy = next(r['accuracy'] for r in metrics if r['method'] == method and r['control'] == 'measured')
        plt.axhline(accuracy, color=color, linestyle=':', label=method)
    plt.axhline(1 / 3, color='red', linestyle=':', label='Chance (1/3)')
    plt.xlabel('Decoder layer output (1-based)')
    plt.ylabel('Accuracy on held-out sets')
    plt.title('Rule probing: held-out images')
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out / 'rule_accuracy.png', dpi=160)
    plt.close()
    print(f'Rule scores saved for {sum(s["split"] == "test" for s in sets)} test sets.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3', 'qwen3_5'), required=True)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    parser.add_argument('--experiment-dir', type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--precision', choices=('4bit', 'fp16', 'bf16', 'fp32'), default='4bit')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--smoke-test', action='store_true', help='Extract two ten-image sets; do not fit probes.')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '01-operator-controls'))
    if args.model == 'qwen3':
        import operator_controls_qwen3 as experiment
    else:
        import operator_controls_qwen35 as experiment
    root = args.experiment_dir.expanduser().resolve()
    if not args.validate_only:
        root = require_results_root(root)
    out = root / 'probing' / args.model / 'rule'
    if args.smoke_test:
        out = out / 'smoke'
    experiment.configure_data(args.data_dir, out)
    if len(experiment.records) == 1000:
        out = out / 'dataset-v1'
    sets = make_sets(experiment)
    report = split_report(sets)
    print(json.dumps(report, indent=2))
    if args.validate_only:
        return
    selected = [sets[0], next(s for s in sets if s['split'] == 'test' and s['rule'] == 'R2')] if args.smoke_test else sets
    digest = hashlib.sha256()
    for record in sorted(experiment.records, key=lambda r: r['filename']):
        digest.update(record['filename'].encode())
        digest.update((experiment.IMAGE_DIR / record['filename']).read_bytes())
    versions = {}
    for package in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'numpy', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = dict(version=1, task='rule_pilot', model=experiment.MODEL_ID, precision=args.precision,
                  metadata_hash=experiment.DATA_HASH, image_hash=digest.hexdigest(), prompt=PROMPT,
                  seed=SEED, pools=list(POOLS), smoke_test=args.smoke_test, sets=selected,
                  split_report=report, software=versions, probe=dict(C=1, max_iter=3000))
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Checkpoint settings differ. Use another existing --experiment-dir; do not mix runs.')
    if not config_path.exists():
        temporary = out / 'config.tmp.json'
        temporary.write_text(json.dumps(config, indent=2))
        temporary.replace(config_path)
    adapter, features = None, []
    random.seed(SEED)
    np.random.seed(SEED)
    for index, scene in enumerate(selected):
        path = out / (scene['set_id'] + '.npz')
        if not path.exists():
            if adapter is None:
                import torch
                torch.manual_seed(SEED)
                if args.model == 'qwen3':
                    if args.upstream is None:
                        parser.error('--upstream is required for Qwen3-VL extraction')
                    experiment.load_upstream_model(experiment.resolve_upstream_code(args.upstream), experiment.MODEL_ID,
                                                   args.precision, 'auto', 384 * 28 * 28)
                else:
                    experiment.load_native_model(experiment.MODEL_ID, args.precision)
                adapter = experiment.vlm
            paths = [str(experiment.IMAGE_DIR / name) for name in scene['image_files']]
            values = capture(adapter, paths, PROMPT, args.model)
            if values.ndim != 3 or values.shape[1] != len(POOLS) or not np.isfinite(values).all():
                raise ValueError('Unexpected or non-finite hidden representations')
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, features=values)
            temporary.replace(path)
        with np.load(path) as saved:
            values = saved['features']
            if values.ndim != 3 or values.shape[1] != len(POOLS) or not np.isfinite(values).all():
                raise ValueError(f'Invalid activation checkpoint: {path}')
            features.append(values)
        print(f'Rule features {index + 1}/{len(selected)}: {scene["set_id"]}')
    if args.smoke_test:
        print('Two-set extraction smoke completed; no probe accuracy measured.')
    else:
        score_rules(np.stack(features), sets, out)


if __name__ == '__main__':
    main()
