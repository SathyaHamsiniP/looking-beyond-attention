"""Validation-only identities and recovery shared by experiments 07/08.

No test-set examples are selected, analyzed, prepared, or scored. Whole-file
hashes identify the supplied source manifest; only validation examples are read.
"""
from collections import Counter
import json
from pathlib import Path
import tempfile

import build_sets as builder
import compare_3_vs_7 as choices
import lora_7rules as seven

NAMES=('07-seven-rule-operator-controls','08-seven-rule-activation-patching')
TRUSTED_INPUT_HELPERS={
    '84a76e09994aacefd68c5dca5b55c850ea078b709bbfaa1dc1a285075ec55c7b',
    'b7d06b07bec4f17e61de70a77d64e03f020904d86c58b35149807de4ee3e3686'}


def local_path(path):
    path=Path(path).resolve()
    if any(part in ('MyDrive','Shareddrives','source-drive','drive') for part in path.parts):
        raise ValueError('New outputs must be local working files, not a mounted Drive')
    repo=Path(__file__).resolve().parents[1]
    if path==repo or repo in path.parents:raise ValueError('Keep experimental artifacts outside Git')
    return path


def validation_view(manifest_path,data):
    m=seven.read(manifest_path)
    # Read only the split tag on non-validation entries; never read their inputs.
    items=[s for s in m['sets'] if s['split']=='validation']
    view=dict(protocol='seven-rule-validation-only-v1',source_manifest_sha256=seven.file_hash(manifest_path),
        source_manifest_canonical_hash=seven.digest(m),rules=m['rules'],sets=items,
        source_metadata_sha256=m['source_metadata_sha256'],image_partitions=m['image_partitions'],
        partition_hash=m['partitions_canonical_hash'],recovery='verified saved manifest')
    validate_view(view,data)
    view['validation_image_sha256']={n:seven.file_hash(Path(data)/'images'/n) for n in sorted({f for s in items for f in s['image_files']})}
    return view


def reconstruct_validation(partition_path,data,expected_metadata_sha256):
    """Existing deterministic builder, validation only; no other sets generated."""
    if seven.file_hash(Path(data)/'metadata.json')!=expected_metadata_sha256:
        raise ValueError('Metadata identity differs; cannot reconstruct')
    assignments=seven.read(partition_path)
    assignments=assignments.get('image_partitions',assignments)
    m=builder.make_manifest(data,seven.RULES_PATH,partition_path,{'validation':7},10,2026)
    view=dict(protocol='seven-rule-validation-only-v1',source_manifest_sha256=None,
        source_manifest_canonical_hash=None,rules=choices.validate_rules(seven.RULES_PATH),sets=m['sets'],
        source_metadata_sha256=expected_metadata_sha256,image_partitions=assignments,partition_hash=seven.digest(assignments),
        recovery='deterministic validation-only reconstruction; full source-manifest identity unavailable')
    validate_view(view,data)
    view['validation_image_sha256']={n:seven.file_hash(Path(data)/'images'/n) for n in sorted({f for s in view['sets'] for f in s['image_files']})}
    return view


def validate_view(view,data):
    spec=choices.validate_rules(seven.RULES_PATH)
    if view['rules']!=spec:raise ValueError('Canonical seven rules differ')
    if seven.file_hash(Path(data)/'metadata.json')!=view['source_metadata_sha256']:
        raise ValueError('Source metadata identity differs')
    metadata=seven.read(Path(data)/'metadata.json')['records'];by_name={r['filename']:r for r in metadata}
    assignments=view['image_partitions']
    if len(by_name)!=1000 or set(assignments)!=set(by_name) or Counter(assignments.values())!={'train':800,'validation':100,'test':100}:
        raise ValueError('Expected existing 1000-image 800/100/100 source partition')
    if seven.digest(assignments)!=view['partition_hash']:raise ValueError('Partition identity differs')
    items=view['sets']
    if len(items)!=49 or Counter(s['rule'] for s in items)!={r:7 for r in choices.IDS}:
        raise ValueError('Expected exactly 49 validation sets, seven per rule')
    identities=set();ids=set()
    for s in items:
        if s['split']!='validation' or s['set_id'] in ids:raise ValueError('Non-validation or duplicate set')
        ids.add(s['set_id']);names=s['image_files'];bits=s['bits']
        if len(names)!=10 or len(set(names))!=10 or len(bits)!=10:raise ValueError('Ten distinct images required')
        identity=tuple(sorted(names))
        if identity in identities:raise ValueError('Duplicate image set')
        identities.add(identity)
        for name,bit in zip(names,bits):
            if Path(name).name!=name or assignments.get(name)!='validation' or name not in by_name:
                raise ValueError('Image outside validation partition')
            if bit!=''.join(map(str,by_name[name]['bits'])) or not (Path(data)/'images'/name).is_file():raise ValueError('Missing/mislabelled validation image')
        table=next(r['true_bits'] for r in spec['rules'] if r['id']==s['rule'])
        if set(bits)!=set(table) or builder.compatible_rules(bits,spec['rules'])!=[s['rule']]:raise ValueError('Incomplete coverage or ambiguous set')
    if 'validation_image_sha256' in view:
        expected={f for s in items for f in s['image_files']}
        if set(view['validation_image_sha256'])!=expected:raise ValueError('Validation image hash inventory differs')
        for name,sha in view['validation_image_sha256'].items():
            if seven.file_hash(Path(data)/'images'/name)!=sha:raise ValueError('Validation image bytes changed: '+name)
    return items


def jobs(view):
    return seven.evaluation_jobs({'sets':view['sets'],'rules':view['rules']},'validation')


def verify_baseline(folder,view,config_name='config.json'):
    folder=Path(folder);config=seven.read(folder/config_name)
    if view['source_manifest_canonical_hash'] is None:raise ValueError('Full source identity unavailable: reconstructed inputs cannot certify an old baseline')
    if config['kind']!='pretrained' or config['adapter'] is not None or config['partition']!='validation' or config['smoke_test']:
        raise ValueError('Need full pretrained validation predictions, never an adapter or smoke run')
    if config['manifest_hash']!=view['source_manifest_canonical_hash'] or config['model']!=seven.MODEL or config['rules']!=view['rules']:
        raise ValueError('Model/rules/source manifest identity mismatch')
    prompts=[choices.prompt_for(view['rules'],choices.mapping(7,r)) for r in range(7)]
    if config['prompts']!=prompts:raise ValueError('Prompt identity mismatch')
    for name in ('compare_3_vs_7.py','pilot_10_vs_20.py','rules_7.json'):
        if config['source_hashes'][name]!=seven.file_hash(Path(__file__).with_name(name)):raise ValueError('Inference source changed: '+name)
    if config['source_hashes']['lora_7rules.py'] not in TRUSTED_INPUT_HELPERS:
        raise ValueError('Unrecognized historical input/scoring helper; audit it before reuse')
    expected=jobs(view);rows=seven.load_rows(folder/'predictions.jsonl',expected)
    if set(rows)!=set(expected):raise ValueError('Incomplete A3: expected all 343 exact validation/rotation keys')
    metrics=seven.metrics(list(rows.values()))
    if (folder/'metrics.json').exists():
        supplied=seven.read(folder/'metrics.json')
        for key in ('sets','evaluations','accuracy','per_rule','confusion','answer_position_counts','rotation_consistency','mean_correct_rule_probability','mean_restricted_answer_nll'):
            if supplied[key]!=metrics[key]:raise ValueError('Saved metrics disagree with raw predictions: '+key)
    tokens=seven.read(folder/'answer_tokens.json') if (folder/'answer_tokens.json').exists() else None
    if tokens is not None and (set(tokens)!=set(seven.LABELS) or len(set(tokens.values()))!=7):raise ValueError('Invalid saved answer tokens')
    return rows,config,tokens


def bootstrap(values,seed=2026,draws=5000):
    import numpy as np
    values=np.asarray(values,dtype=float)
    if not len(values):raise ValueError('No clusters')
    sampled=np.random.default_rng(seed).integers(0,len(values),size=(draws,len(values)))
    return np.quantile(values[sampled].mean(axis=1),[.025,.975]).tolist()
