"""Select a decoder block on validation and patch it on reserved test pairs."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
from importlib.metadata import version
import math
from pathlib import Path
from statistics import mean
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import BENCHMARK, source_path
import activation_patching_7rules as patch
from seven_rule_validation import seven, choices, bootstrap

RESULTS = BENCHMARK / 'results/seven-rules/qwen3.5-4b'
LORA = RESULTS / 'lora'
BLOCKS = 32
TOL = 1e-5


def confirmation_pairs(manifest):
    """Use each test set as receiver once. Balance donor rules and minimize overlap."""
    groups = {rule: sorted((s for s in manifest['sets'] if s['split'] == 'test'
                           and s['rule'] == rule), key=lambda s: s['set_id'])
              for rule in choices.IDS}
    counts = Counter()
    pairs = []
    for index, rule in enumerate(choices.IDS):
        for j, receiver in enumerate(groups[rule]):
            other = choices.IDS[(index + 1 + j % 6) % 7]
            for condition, target in [('same', rule), ('different', other)]:
                candidates = [s for s in groups[target] if s != receiver]
                images = set(receiver['image_files'])
                def priority(s):
                    return (len(images.intersection(s['image_files'])), counts[s['set_id']], s['set_id'])
                donor = min(candidates, key=priority)
                overlap = priority(donor)[0]
                sid, did = receiver['set_id'], donor['set_id']
                counts[did] += 1
                pairs.append(dict(pair_id=f'{sid}:{condition}', condition=condition,
                                  control_group=sid, receiver_set=sid, receiver_rule=rule,
                                  donor_set=did, donor_rule=target, shared_images=overlap))
    return pairs


def audit(manifest):
    provenance = seven.read(LORA / 'provenance.json')
    if seven.digest(manifest) != provenance['dataset']['canonical_hash']:
        raise ValueError('Manifest differs from the completed seven-rule run')
    jobs = seven.evaluation_jobs(manifest, 'validation')
    before = seven.load_rows(RESULTS / 'operator-controls/predictions.jsonl', jobs)
    after = seven.load_rows(LORA / 'finetuned-validation/predictions.jsonl', jobs)
    if set(before) != set(jobs) or set(after) != set(jobs):
        raise ValueError('Incomplete validation predictions')
    transitions = Counter((bool(before[k]['correct']), bool(after[k]['correct'])) for k in jobs)
    pairs = confirmation_pairs(manifest)
    return dict(manifest_hash=seven.digest(manifest), validation_inputs=len(jobs),
                both_correct=transitions[True, True], corrected=transitions[False, True],
                lost=transitions[True, False], both_wrong=transitions[False, False],
                selection_forwards=len(jobs)*(BLOCKS+1), test_pairs=len(pairs),
                test_pairs_with_shared_images=sum(p['shared_images'] > 0 for p in pairs),
                test_patches_per_model=len(pairs)*7, adapter_identity=seven.read(
                    LORA / 'training/complete.json')['adapter_identity'], pairs=pairs), after


@contextmanager
def silence(block_modules):
    """Zero only this block's additive LoRA term. Restore it even on a failed forward."""
    previous = []
    try:
        for module in block_modules:
            previous.append((module, module.scaling['default']))
            module.scaling['default'] = 0.0
        yield
    finally:
        for module, scale in previous:
            module.scaling['default'] = scale


def rank_layers(rows):
    by_layer = [defaultdict(list) for _ in range(BLOCKS)]
    rules = {}
    for row in rows:
        rules[row['set_id']] = row['true_rule']
        for group, delta in zip(by_layer, row['loss_increases'], strict=True):
            group[row['set_id']].append(delta)
    table = []
    for layer, group in enumerate(by_layer, 1):
        scores = {sid: mean(values) for sid, values in group.items()}
        table.append(dict(layer=layer, mean_loss_increase=mean(scores.values()),
                          set_bootstrap_ci=bootstrap(list(scores.values())),
                          per_rule={r: mean(v for s,v in scores.items() if rules[s] == r)
                                    for r in sorted(set(rules.values()))}))
    return sorted(table, key=lambda x: (-x['mean_loss_increase'], x['layer']))


def check_baseline(measured, expected):
    if set(measured) != set(expected) or any(
            not math.isfinite(v) or abs(v-expected[k]) > TOL for k,v in measured.items()):
        raise ValueError('Unpatched probabilities differ from the reference')


def select(model, processor, tokens, blocks, manifest, data, reference, out):
    import torch
    from peft.tuners.lora.layer import LoraLayer
    modules = [[m for m in block.modules() if isinstance(m, LoraLayer)] for block in blocks]
    target_names = seven.read(LORA / 'training/trainable.json')['target_modules']
    base = model.get_base_model()
    if {id(m) for group in modules for m in group} != {id(base.get_submodule(n)) for n in target_names}:
        raise ValueError('Loaded LoRA module inventory differs from training')
    if any(not group for group in modules) or any(m.merged or m.disable_adapters or
            m.active_adapters != ['default'] for group in modules for m in group):
        raise ValueError('Expected active, unmerged LoRA modules in each block')
    jobs = seven.evaluation_jobs(manifest, 'validation')
    rows = []
    for key, job in jobs.items():
        path = out / (key.replace(':', '_')+'.json')
        if not path.exists():
            inputs = seven.prepare(model, processor, job, data, job['prompt'], tokens)
            target = tokens[seven.LABELS.index(job['true_label'])]
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0,-1].float()
                probs = logits[tokens].softmax(0).cpu().tolist()
                check_baseline(dict(zip(seven.LABELS, probs)), reference[key]['probabilities'])
                loss = -logits.log_softmax(0)[target].item()
                increases = []
                for group in modules:
                    with silence(group):
                        increases.append(seven.answer_loss(model, inputs, target).item()-loss)
            seven.atomic_json(path, dict(key=key, set_id=job['set_id'], true_rule=job['true_rule'],
                                        full_loss=loss, loss_increases=increases))
        row = seven.read(path)
        if (row['key'], row['set_id'], row['true_rule']) != (key,job['set_id'],job['true_rule']) or (
                len(row['loss_increases']) != BLOCKS or not all(map(math.isfinite,row['loss_increases']))):
            raise ValueError(f'Invalid saved ablation for {key}')
        rows.append(row)
        if len(rows) % 7 == 0:
            print(f'Validation inputs complete: {len(rows)}/{len(jobs)}', flush=True)
    ranking = rank_layers(rows)
    seven.atomic_json(out/'ranking.json', ranking)
    if ranking[0]['mean_loss_increase'] <= 0:
        raise ValueError('No block had a positive mean loss increase. Inspect ranking.json.')
    return ranking[0]['layer']


def confirm(model, processor, tokens, blocks, layer, manifest, data, pairs, out):
    engine = patch.intervention_engine(tokens)
    wrapper = SimpleNamespace(model=model)
    jobs = seven.evaluation_jobs(manifest, 'test')
    block = blocks[layer-1]

    @lru_cache(maxsize=32)
    def capture(sid, rotation):
        job = jobs[f'{sid}:{rotation}']
        inputs = seven.prepare(model, processor, job, data, job['prompt'], tokens)
        # A one-element block list gives the selected vector capture index 1.
        return engine.record_original(wrapper, inputs, job['option_mapping'], [block])

    results = {}
    for pair in pairs:
        sid = pair['receiver_set']
        path = out / (pair['pair_id'].replace(':','_')+'.json')
        if not path.exists():
            rows = []
            for rotation in range(7):
                job = jobs[f'{sid}:{rotation}']
                receiver = capture(sid, rotation)
                donor = capture(pair['donor_set'], rotation)
                inputs = seven.prepare(model, processor, job, data, job['prompt'], tokens)
                def replace(original):
                    return engine.patched_probabilities(wrapper, inputs, job['option_mapping'],
                                                        block, original['activations'][1]['final'], 'final')
                if rotation == 0:
                    check_baseline(replace(receiver), receiver['probabilities'])
                rows.append(patch.effect(pair, rotation, layer, 'final', receiver['probabilities'],
                                         donor['probabilities'], replace(donor)))
            seven.atomic_json(path, rows)
        rows = seven.read(path)
        if len(rows) != 7:
            raise ValueError('Incomplete confirmation group')
        for rotation, row in enumerate(rows):
            if row != patch.effect(pair, rotation, layer, 'final', row['before_probabilities'],
                                   row['donor_probabilities'], row['patched_probabilities']):
                raise ValueError('Invalid saved intervention')
            results[(pair['pair_id'], rotation)] = row
    capture.cache_clear()
    return results


def compare(before, after):
    if before.keys() != after.keys():
        raise ValueError('Patch measurements use different pairs or rotations')
    matched = defaultdict(dict)
    by_condition = defaultdict(list)
    wrong_donor = {'base': [], 'lora': []}
    by_donor_rule = defaultdict(list)
    shared = set()
    for key, row in after.items():
        base = before[key]
        effect = row['delta_true_rule']-base['delta_true_rule']
        matched[(row['receiver_set'], row['receiver_rule'],row['rotation'])][row['condition']] = effect
        by_condition[row['condition']].append(effect)
        if row['shared_images']:
            shared.add(row['receiver_set'])
        if row['condition'] == 'different':
            by_donor_rule[row['donor_rule']].append(effect)
        for name, trial in [('base',base),('lora',row)]:
            if trial['condition']=='different' and trial['donor_predicted_rule'] != trial['donor_true_rule']:
                wrong_donor[name].append(trial['delta_true_rule']-trial['delta_donor_answer'])
    receivers = defaultdict(list)
    for (sid, rule, rotation), group in matched.items():
        receivers[rule,sid].append(group['different']-group['same'])
    values = {k:mean(v) for k,v in receivers.items()}
    disjoint = [v for (_, sid),v in values.items() if sid not in shared]
    return dict(disjoint_receiver_count=len(disjoint),
                disjoint_control_adjusted_mean=mean(disjoint) if disjoint else None,
                different_pairs_by_donor_rule={r:mean(v) for r,v in by_donor_rule.items()},
                mean_control_adjusted_lora_minus_base=mean(values.values()),
                receiver_bootstrap_ci=bootstrap(list(values.values())),
                per_receiver_rule={r:mean(v for (rule,_),v in values.items() if rule==r) for r in choices.IDS},
                lora_minus_base_by_condition={k:mean(v) for k,v in by_condition.items()},
                wrong_donor_true_minus_predicted_shift={k:dict(count=len(v),mean=mean(v) if v else None)
                                                       for k,v in wrong_donor.items()})


def run(args, manifest, plan, reference):
    from peft import PeftModel
    import torch
    from contextlib import nullcontext
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA runtime is required for model measurements')
    seven.validate_manifest(manifest, args.data_dir)
    if seven.adapter_identity(args.adapter) != plan['adapter_identity']:
        raise ValueError('Adapter differs from the selected seven-rule epoch-3 checkpoint')
    expected = seven.read(LORA/'training/config.json')['software']
    if any(version(name) != value for name,value in expected.items()):
        raise ValueError('Use the software versions saved in lora/training/config.json')
    base, processor, _ = choices.shared.load_model()
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    tokens = choices.label_tokens(processor.tokenizer)
    if dict(zip(seven.LABELS, tokens)) != seven.read(LORA/'training/answer_tokens.json'):
        raise ValueError('Answer-token identity mismatch')
    blocks = patch.intervention_engine(tokens).decoder_layers(model.get_base_model())
    if len(blocks) != BLOCKS:
        raise ValueError('Unexpected decoder block count')
    validation_jobs = seven.evaluation_jobs(manifest, 'validation')
    original = seven.load_rows(RESULTS/'operator-controls/predictions.jsonl', validation_jobs)
    engine = patch.intervention_engine(tokens)
    with model.disable_adapter():
        for rule in choices.IDS:
            job = next(j for j in validation_jobs.values() if j['true_rule'] == rule)
            inputs = seven.prepare(model, processor, job, args.data_dir, job['prompt'], tokens)
            measured = engine.probabilities(model, inputs, job['option_mapping'])
            expected_probs = original[job['key']]['probabilities']
            check_baseline(measured, {job['option_mapping'][k]:v for k,v in expected_probs.items()})
    config = dict(protocol='validation-loss-test-final-token-v1', plan_hash=seven.digest(plan),
                  runner=seven.file_hash(Path(__file__)), software=expected,
                  source_hashes={name:seven.file_hash(source_path(name)) for name in
                                 ('activation_patching_7rules.py','lora_7rules.py','compare_3_vs_7.py',
                                  'seven_rule_validation.py','pilot_10_vs_20.py')},
                  hook_hash=seven.file_hash(patch.HISTORICAL))
    with seven.output(args.out, config, args.resume):
        for name in ('selection','base','lora'):
            (args.out/name).mkdir(exist_ok=True)
        seven.atomic_json(args.out/'plan.json', plan)
        layer = select(model, processor, tokens, blocks, manifest, args.data_dir, reference, args.out/'selection')
        print(f'Selected decoder block {layer}. Running matched test pairs.', flush=True)
        variants = {}
        for name in ('base','lora'):
            with model.disable_adapter() if name == 'base' else nullcontext():
                variants[name] = confirm(model, processor, tokens, blocks, layer, manifest,
                                         args.data_dir, plan['pairs'], args.out/name)
        report = compare(variants['base'], variants['lora'])
        seven.atomic_json(args.out/'comparison.json', dict(selected_layer=layer, **report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit','run'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = seven.read(args.manifest)
    plan, reference = audit(manifest)
    if args.action == 'audit':
        seven.atomic_json(args.out, plan)
        print({k:v for k,v in plan.items() if k not in ('pairs','adapter_identity')})
        return
    if not args.data_dir or not args.adapter:
        parser.error('run requires --data-dir and --adapter')
    run(args, manifest, plan, reference)


if __name__ == '__main__':
    main()
