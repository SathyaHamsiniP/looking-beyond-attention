"""Four-slot positive/negative evidence-integration pilot for Qwen3.5."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '01-operator-controls'))
import operator_controls_qwen35 as experiment
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root
from probe_rules import write_csv

PAIRS = {'R1': ('010', '100'), 'R2': ('001', '100'), 'R3': ('001', '010')}
CONDITIONS = ('first_only', 'second_only', 'both')


def candidates(examples):
    return [rule for rule in experiment.RULES
            if all(bool(experiment.satisfies(rule, e['bits'])) == e['positive'] for e in examples)]


def pattern(examples):
    # A bag of labelled scene types ignores image positions and visual appearance.
    return tuple(sorted(Counter((''.join(map(str, e['bits'])), e['positive']) for e in examples).items()))


def make_dataset(records):
    lookup = {(''.join(map(str, r['bits'])), r['variation']): r['filename'] for r in records}
    if len(lookup) != 48 or set(lookup) != {( ''.join(map(str, bits)), v) for bits in product((0, 1), repeat=3) for v in range(6)}:
        raise ValueError('Expected the original eight configurations with six variations each.')

    def example(bits, variation, positive):
        return dict(filename=lookup[bits, variation], bits=list(map(int, bits)), positive=positive)

    families = []
    for split, blocks in [('train', [(0, 1), (2, 3)]), ('test', [(4, 5)])]:
        for block, (first, second) in enumerate(blocks):
            for rule, pair in PAIRS.items():
                diagnostic = [example(bits, first, False) for bits in pair]
                replacement = example('000', first, False)
                common = [example('111', second, True), example('000', second, False)]
                conditions = {'first_only': [diagnostic[0], replacement] + common,
                              'second_only': [replacement, diagnostic[1]] + common,
                              'both': diagnostic + common}
                families.append(dict(family_id=f'{split}_{block}_{rule}', split=split, rule=rule,
                                     diagnostic=diagnostic, conditions=conditions))
    validate(families)
    return families


def validate(families):
    identities = set()
    for family in families:
        rule = family['rule']
        diagnostic = family['diagnostic']
        assert all(len(candidates([e])) == 2 for e in diagnostic)
        assert candidates(diagnostic) == [rule]
        for name, examples in family['conditions'].items():
            assert len(examples) == len({e['filename'] for e in examples}) == 4
            assert sum(e['positive'] for e in examples) == 1
            assert all(bool(experiment.satisfies(rule, e['bits'])) == e['positive'] for e in examples)
            assert rule in candidates(examples)
            assert len(candidates(examples)) == (1 if name == 'both' else 2)
        assert candidates(family['conditions']['both']) == [rule]
        assert candidates(family['conditions']['both'][2:]) == list(experiment.RULES)
        jobs = presentations(family)
        assert len(jobs) == 36
        for condition in CONDITIONS:
            subset = [j for j in jobs if j['condition'] == condition]
            for filename in {e['filename'] for e in family['conditions'][condition]}:
                assert Counter(i for j in subset for i, e in enumerate(j['examples'])
                               if e['filename'] == filename) == Counter({i: 3 for i in range(4)})
            for candidate in experiment.RULES:
                assert Counter(label for j in subset for label, r in j['mapping'].items()
                               if r == candidate) == Counter({label: 4 for label in experiment.LABELS})
        identity = tuple(sorted(e['filename'] for e in family['conditions']['both']))
        assert identity not in identities
        identities.add(identity)
    images = {split: {e['filename'] for f in families if f['split'] == split
                      for examples in f['conditions'].values() for e in examples} for split in ('train', 'test')}
    assert images['train'].isdisjoint(images['test'])


def presentations(family):
    jobs = []
    # Each slot and each answer label sees each role equally often.
    for condition in CONDITIONS:
        examples = family['conditions'][condition]
        for rotation in range(4):
            ordered = examples[rotation:] + examples[:rotation]
            for layout, (mapping, display) in enumerate(experiment.layouts('cyclic')):
                jobs.append(dict(job_id=f'{family["family_id"]}_{condition}_{rotation}_{layout}',
                                 family_id=family['family_id'], rule=family['rule'], condition=condition,
                                 rotation=rotation, layout=layout, examples=ordered,
                                 mapping=mapping, display=list(display)))
    return jobs


def prompt_for(job):
    labels = '\n'.join(f'Image {i}: {"positive" if e["positive"] else "negative"}'
                       for i, e in enumerate(job['examples'], 1))
    choices = '\n'.join(f'{label}: {experiment.EXPRESSIONS[job["mapping"][label]]}' for label in job['display'])
    return f'''Find the Boolean rule consistent with all four labelled images.
A is true when a green cube is present. B is true when a blue sphere is present.
C is true when a red cylinder is present. Each concept requires both colour and shape.
Other objects are irrelevant. Positive means the rule is true; negative means it is false.
The image labels, in order, are:
{labels}
Candidate rules:
{choices}
AND requires both inputs. OR requires at least one input. Follow the parentheses.
If several candidates are consistent, choose among them.
Reply with nothing but X, Y, or Z.
ANSWER:'''


def baseline_table(families):
    table = defaultdict(Counter)
    for family in families:
        if family['split'] == 'train':
            for examples in family['conditions'].values():
                table[pattern(examples)][family['rule']] += 1
    return table


def baseline_probs(job, table):
    compatible = candidates(job['examples'])
    symbolic = {r: float(r in compatible) / len(compatible) for r in experiment.RULES}
    counts = table.get(pattern(job['examples']), Counter())
    total = sum(counts.values())
    scene_pattern = {r: counts[r] / total if total else 1 / 3 for r in experiment.RULES}
    return {'symbolic': symbolic, 'scene_pattern': scene_pattern}


def report(families):
    result = dict(evaluation='small image-disjoint pilot; families share images within partitions',
                  slots=4, positive_slots=1, negative_slots=3, chance=1 / 3,
                  presentations_per_family=36, training='metadata scene-pattern baseline only; model is frozen')
    for split in ('train', 'test'):
        subset = [f for f in families if f['split'] == split]
        images = {e['filename'] for f in subset for examples in f['conditions'].values() for e in examples}
        result[split] = dict(families=len(subset), unique_images=len(images), conditions_per_family=3,
                             max_mutually_image_disjoint_families=len(subset) // 3)
    return result


def summarize(rows, out):
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['family_id'], row['condition'], row['truth']].append(row)
    grouped = []
    for (method, family, condition, truth), values in groups.items():
        probabilities = {r: float(np.mean([v[f'probability_{r}'] for v in values])) for r in experiment.RULES}
        best = max(probabilities.values())
        winners = [r for r, p in probabilities.items() if abs(p - best) < 1e-8]
        grouped.append(dict(method=method, family_id=family, condition=condition, truth=truth,
                            presentations=len(values), true_rule_probability=probabilities[truth],
                            # Ties get fractional credit instead of favouring the first rule.
                            selection_accuracy=float(truth in winners) / len(winners),
                            mean_presentation_accuracy=float(np.mean([v['selection_accuracy'] for v in values])),
                            consistent_probability=float(np.mean([v['consistent_probability'] for v in values]))))
    write_csv(out / 'family_summary.csv', grouped)
    paired = []
    for method, family, truth in sorted({(r['method'], r['family_id'], r['truth']) for r in grouped}):
        scores = {r['condition']: r['true_rule_probability'] for r in grouped if r['method'] == method and r['family_id'] == family}
        paired.append(dict(method=method, family_id=family, truth=truth, both_minus_first=scores['both'] - scores['first_only'],
                           both_minus_second=scores['both'] - scores['second_only']))
    write_csv(out / 'paired_changes.csv', paired)
    summary = []
    for method in sorted({r['method'] for r in grouped}):
        for condition in CONDITIONS:
            values = [r for r in grouped if r['method'] == method and r['condition'] == condition]
            summary.append(dict(method=method, condition=condition, families=len(values),
                                selection_accuracy=float(np.mean([r['selection_accuracy'] for r in values])),
                                true_rule_probability=float(np.mean([r['true_rule_probability'] for r in values]))))
    write_csv(out / 'condition_summary.csv', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    parser.add_argument('--experiment-dir', type=Path, default=DEFAULT_RESULTS_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--generate-only', action='store_true', help='Save validated pilot design without loading a model.')
    mode.add_argument('--smoke-test', action='store_true', help='Run three matched conditions for one family/layout/order.')
    args = parser.parse_args()
    root = args.experiment_dir.expanduser().resolve()
    if not args.validate_only:
        root = require_results_root(root)
    out = root / 'probing/qwen3_5/rule/evidence_integration'
    if args.smoke_test:
        out = out / 'smoke'
    experiment.configure_data(args.data_dir, out)
    families = make_dataset(experiment.records)
    print(json.dumps(report(families), indent=2))
    if args.validate_only:
        return
    jobs = [job for family in families if family['split'] == 'test' for job in presentations(family)]
    if args.smoke_test:
        jobs = [j for j in jobs if j['family_id'] == 'test_0_R1' and j['rotation'] == j['layout'] == 0]
    digest = hashlib.sha256()
    for r in sorted(experiment.records, key=lambda r: r['filename']):
        digest.update(r['filename'].encode())
        digest.update((experiment.IMAGE_DIR / r['filename']).read_bytes())
    versions = {}
    for package in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = dict(version=1, seed=2026, software=versions, model=experiment.MODEL_ID, precision='4bit', metadata_hash=experiment.DATA_HASH,
                  image_hash=digest.hexdigest(), report=report(families), families=families,
                  jobs=jobs, prompts=[prompt_for(j) for j in jobs], smoke_test=args.smoke_test)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Existing experiment configuration differs; do not mix runs.')
    temporary = out / 'config.tmp.json'
    temporary.write_text(json.dumps(config, indent=2))
    temporary.replace(path)
    if args.generate_only:
        print('Validated design saved; no model was loaded.')
        return
    table = baseline_table(families)
    checkpoint = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()] if checkpoint.exists() else []
    done = {row['job_id'] for row in rows if row['method'] == 'qwen3_5'}
    valid = {j['job_id'] for j in jobs}
    if not done <= valid:
        raise ValueError('Checkpoint contains jobs outside this configuration.')
    model_loaded = False
    for index, job in enumerate(jobs):
        if job['job_id'] not in done:
            if not model_loaded:
                import torch
                torch.manual_seed(2026)
                experiment.load_native_model(experiment.MODEL_ID, '4bit')
                model_loaded = True
            paths = [str(experiment.IMAGE_DIR / e['filename']) for e in job['examples']]
            raw = experiment.vlm.option_probs(paths, prompt_for(job), experiment.TOKEN_IDS)
            probabilities = {job['mapping'][label]: float(raw[label]) for label in experiment.LABELS}
            if (not all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
                    or not np.isclose(sum(probabilities.values()), 1)):
                raise ValueError('Expected normalized, finite candidate probabilities.')
            results = dict(qwen3_5=probabilities, **baseline_probs(job, table))
            new = []
            compatible = candidates(job['examples'])
            for method, probabilities in results.items():
                best = max(probabilities.values())
                winners = [r for r, p in probabilities.items() if abs(p - best) < 1e-8]
                row = dict(job_id=job['job_id'], family_id=job['family_id'], condition=job['condition'],
                           rotation=job['rotation'], layout=job['layout'], method=method, truth=job['rule'],
                           selected_rules='|'.join(winners), selection_accuracy=float(job['rule'] in winners) / len(winners),
                           consistent_probability=sum(probabilities[r] for r in compatible))
                row.update({f'probability_{r}': probabilities[r] for r in experiment.RULES})
                new.append(row)
            # Rewrite atomically so interruption cannot leave half a presentation.
            rows.extend(new)
            temporary = out / 'predictions.tmp.jsonl'
            temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            temporary.replace(checkpoint)
            done.add(job['job_id'])
        print(f'Evidence presentations {index + 1}/{len(jobs)}')
    write_csv(out / 'predictions.csv', rows)
    summarize(rows, out)
    print('Small pilot completed. Single-example conditions are deliberately ambiguous.')


if __name__ == '__main__':
    main()
