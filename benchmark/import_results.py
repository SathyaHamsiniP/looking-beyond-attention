"""Copy compact artifacts into model/rule-specific result trees, never move them.

Exact local source folders only. Configurations must identify Qwen3.5-4B;
images, features, adapters, optimizer states and dataset manifests are excluded.
Experiment-specific numerical verification is recorded in provenance by callers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import lora_7rules as seven

EXPERIMENTS=('operator-controls','probing','activation-patching','lora')
ALLOWED={'.json','.jsonl','.csv','.png','.md','.txt'}
EXCLUDE={'manifest.json','partitions.json','validation_view.json','selection_freeze.json','paired_sets.json','input_manifest.json','trainable.json'}


def identified_model(config):
    value=config.get('model',config.get('settings',{}).get('model'))
    if isinstance(value,dict):value=value.get('model')
    return value


def import_folder(source,destination,rule_count,experiment,source_url=None,verification=None):
    source=Path(source).resolve();destination=Path(destination)
    resolved=destination.resolve()
    if source==resolved or source in resolved.parents or resolved in source.parents:raise ValueError('Source and destination must not overlap')
    if rule_count not in (3,7) or experiment not in EXPERIMENTS:raise ValueError('Unknown task/model destination')
    configs=[p for p in source.rglob('*config.json') if not any(part in ('features','best_adapter','latest','latest.tmp','smoke') for part in p.relative_to(source).parts)]
    if not configs or not any(identified_model(seven.read(p))=='Qwen/Qwen3.5-4B' for p in configs):
        raise ValueError('No Qwen3.5-4B model identity in source configs')
    for p in configs:
        model=identified_model(seven.read(p))
        if model is not None and model!='Qwen/Qwen3.5-4B':raise ValueError('Do not mix Qwen3-VL/other models with Qwen3.5-4B')
        rules=seven.read(p).get('rules')
        generating=3 if seven.read(p).get('protocol')=='original-3-under-3-vs-7-v1' else len(rules['rules']) if isinstance(rules,dict) and 'rules' in rules else None
        if generating is not None and generating!=rule_count:raise ValueError('Generating-rule count differs from destination')
    expected_segment='three-rules' if rule_count==3 else 'seven-rules'
    if expected_segment not in destination.parts or 'qwen3.5-4b' not in destination.parts or experiment not in destination.parts:
        raise ValueError('Destination must identify rule family, Qwen3.5-4B and experiment')
    receipts=[]
    for path in sorted(source.rglob('*')):
        relative=path.relative_to(source)
        if not path.is_file() or path.is_symlink() or any(x.startswith('.') or x in ('features','best_adapter','latest','latest.tmp','smoke') for x in relative.parts):continue
        if path.name=='selection.json' and experiment!='lora':continue
        if path.suffix.lower() not in ALLOWED or path.name in EXCLUDE or path.name.startswith(('tokenizer','processor')):continue
        if path.stat().st_size>2*1024*1024:raise ValueError('Unexpectedly large compact result: '+str(relative))
        if path.name=='provenance.json':continue
        target=destination/relative
        if target.exists() and seven.file_hash(target)!=seven.file_hash(path):raise ValueError('Refusing to overwrite a different imported artifact')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        receipts.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=seven.file_hash(path)))
    provenance=dict(source_url=source_url,source_folder=str(source),rule_count=rule_count,model='Qwen/Qwen3.5-4B',
        experiment=experiment,verification=verification or ['Model identity verified; metric-level verification not supplied'],files=receipts,
        policy='Copied byte-for-byte, originals unchanged. No datasets, adapters, activation arrays or optimizer states imported.')
    seven.atomic_json(destination/'provenance.json',provenance)
    print(json.dumps(dict(destination=str(destination),files=len(receipts),source=source_url or str(source)),indent=2),flush=True)
    return provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--rule-count',type=int,choices=(3,7),required=True);p.add_argument('--experiment',choices=EXPERIMENTS,required=True);p.add_argument('--source-url')
    a=p.parse_args();import_folder(a.source,a.destination,a.rule_count,a.experiment,a.source_url)


if __name__=='__main__':main()
