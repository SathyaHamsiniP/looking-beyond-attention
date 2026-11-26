"""09: frozen seven-rule probes, exact saved train/validation sets, 1/5/10 images.

One image is an ambiguous control. Five images are oracle-selected identifying
subsets, not regenerated sets. Test examples are never selected or scored.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import importlib.util
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import REPO, source_path

import shutil

import numpy as np
import compare_3_vs_7 as choices
import lora_7rules as seven
import seven_rule_validation as validation

CONDITIONS=(1,5,10)
SEED=2026
FEATURE_SOURCES=('benchmark_paths.py','probing_7rules.py','probing.py','compare_3_vs_7.py','rules_7.json')
INFERENCE_PACKAGES={'torch':'2.14.1+cu130','transformers':'5.22.1','bitsandbytes':'0.50.2','accelerate':'1.15.0','pillow':'13.0.0'}


def capture_helpers():
    # Isolated historical benchmark capture and atomic feature IO. Only readout
    # changes to the existing seven-label scorer; the three-rule runner is untouched.
    spec=importlib.util.spec_from_file_location('seven_probe_feature_helpers',source_path('probing.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.evaluate=choices.evaluate_numeric
    return module


def minima(spec):
    report={}
    for rule in spec['rules']:
        for n in range(1,len(rule['true_bits'])+1):
            witnesses=[list(bits) for bits in combinations(rule['true_bits'],n)
                       if seven.builder.compatible_rules(bits,spec['rules'])==[rule['id']]]
            if witnesses:
                report[rule['id']]=dict(minimum_distinct_configurations=n,witness=witnesses[0]);break
        if report[rule['id']]['minimum_distinct_configurations']!=5:
            raise ValueError('Canonical seven-rule identifiability minimum is not five')
    return report


def select_one(names):
    return min(range(len(names)),key=lambda i:hashlib.sha256(f'{SEED}|{names[i]}'.encode()).digest())


def subsets(item,spec):
    names=item['image_files'];bits=item['bits']
    if len(names)!=10 or len(set(names))!=10 or len(bits)!=10:raise ValueError('Expected unchanged ten-image source set')
    if seven.builder.compatible_rules(bits,spec['rules'])!=[item['rule']]:raise ValueError('Source ten-image set is not uniquely identifying')
    # One-image selection does not use the generating label or Boolean bits.
    first=select_one(names)
    consistent=seven.builder.compatible_rules([bits[first]],spec['rules'])
    if len(consistent)<2:raise ValueError('One-image control must be ambiguous')
    five=None
    for indices in combinations(range(10),5):
        observed=[bits[i] for i in indices]
        if len(set(observed))==5 and seven.builder.compatible_rules(observed,spec['rules'])==[item['rule']]:
            five=indices;break
    if five is None:raise ValueError(f"No identifying five-image subset in {item['set_id']}")
    return {str(n):dict(indices=list(indices),image_files=[names[i] for i in indices],bits=[bits[i] for i in indices],
        compatible_rules=seven.builder.compatible_rules([bits[i] for i in indices],spec['rules']),
        method='label-blind SHA256 filename selection; ambiguous control' if n==1 else
               'oracle: lexicographically first five original indices with five distinct identifying configurations' if n==5 else 'unchanged full original ordered set')
        for n,indices in ((1,[first]),(5,five),(10,range(10)))}


def plan_inputs(manifest_path,data):
    m=seven.read(manifest_path);spec=choices.validate_rules(seven.RULES_PATH)
    if m['rules']!=spec or m['rules_sha256']!=seven.file_hash(seven.RULES_PATH):raise ValueError('Canonical rules differ')
    if seven.file_hash(Path(data)/'metadata.json')!=m['source_metadata_sha256']:raise ValueError('Source metadata changed')
    records=seven.read(Path(data)/'metadata.json')['records'];by_name={r['filename']:r for r in records}
    assignments=m['image_partitions']
    if len(by_name)!=1000 or set(assignments)!=set(by_name) or Counter(assignments.values())!={'train':800,'validation':100,'test':100}:
        raise ValueError('Original 800/100/100 source-image split required')
    if seven.digest(assignments)!=m['partitions_canonical_hash']:raise ValueError('Partition mapping changed')
    items=[s for s in m['sets'] if s['split'] in ('train','validation')]
    if Counter((s['split'],s['rule']) for s in items)!={(split,r):n for split,n in (('train',21),('validation',7)) for r in choices.IDS}:
        raise ValueError('Expected the saved 147 training and 49 validation sets')
    if len({s['set_id'] for s in items})!=196:raise ValueError('Duplicate source set IDs')
    used=defaultdict(Counter);selected={};hashes={};identities=set()
    for item in items:
        sid=item['set_id']
        if not sid or Path(sid).name!=sid:raise ValueError('Unsafe set ID')
        identity=tuple(sorted(item['image_files']))
        if identity in identities:raise ValueError('Duplicate source set identity')
        identities.add(identity)
        for name,bit in zip(item['image_files'],item['bits']):
            if Path(name).name!=name or assignments.get(name)!=item['split'] or name not in by_name or bit!=''.join(map(str,by_name[name]['bits'])):
                raise ValueError('Image leakage or metadata mismatch')
            if not (Path(data)/'images'/name).is_file():raise ValueError('Missing source image: '+name)
            if name not in hashes:hashes[name]=seven.file_hash(Path(data)/'images'/name)
        used[item['split']].update(item['image_files']);selected[sid]=subsets(item,spec)
    if set(used['train'])&set(used['validation']):raise ValueError('Train/validation source-image leakage')
    report=dict(source_manifest_sha256=seven.file_hash(manifest_path),source_manifest_canonical_hash=seven.digest(m),
        metadata_sha256=m['source_metadata_sha256'],partition_hash=m['partitions_canonical_hash'],
        source_sets_hash=seven.digest(items),rules=spec,minima=minima(spec),selection=selected,
        image_sha256=hashes,split_statistics={split:dict(sets=sum(s['split']==split for s in items),
            unique_source_images=len(counts),reused_images=sum(v>1 for v in counts.values())) for split,counts in used.items()},
        limitation='One image is an ambiguous generating-label control. Five-image selection uses oracle knowledge of the true rule and truth tables. Not the older three-rule uniquely diagnostic one-image protocol. Test examples are excluded.')
    return items,report


def prompt(spec,n,rotation):
    return choices.prompt_for(spec,choices.mapping(7,rotation)).replace('Images 1-10','Images 1-'+str(n),1)


def feature_dir(out,sid,n):return Path(out)/'features'/f'{sid}_n{n}'


def feature_record(folder,expected,helpers):
    row=seven.read(folder/'metadata.json')
    for key,value in expected.items():
        if row[key]!=value:raise ValueError('Saved feature input identity differs: '+key)
    path=folder/'features.npz'
    if seven.file_hash(path)!=row['feature_sha256']:raise ValueError('Feature checksum differs')
    values=helpers.load_features(path)
    if values.shape[0]!=32 or values.shape[1]<1:raise ValueError('Expected all 32 decoder layers')
    if row['input_tokens']<1:raise ValueError('Invalid saved token count')
    with np.load(path) as archive:
        if int(archive['input_tokens'])!=row['input_tokens']:raise ValueError('Saved token count differs from feature metadata')
    probs=row['probabilities']
    if set(probs)!=set(seven.LABELS) or not np.isfinite(list(probs.values())).all() or any(v<0 or v>1 for v in probs.values()) or not np.isclose(sum(probs.values()),1):
        raise ValueError('Invalid saved model probabilities')
    if 'rotation' in expected:
        label=max(seven.LABELS,key=probs.get);mapping=choices.mapping(7,expected['rotation'])
        if row['predicted_label']!=label or row['predicted_rule']!=mapping[label] or row['correct']!=(mapping[label]==expected['true_rule']):
            raise ValueError('Saved prediction disagrees with probabilities')
    return row,values


def extract(out,data,items,report,rotation,resume=False,smoke=False):
    out=validation.local_path(out);helpers=capture_helpers()
    work=items[:1] if smoke else items
    cfg=dict(protocol='seven-rule-probes-1-5-10-v1',source_manifest=report['source_manifest_canonical_hash'],
        source_sets_hash=report['source_sets_hash'],selection_hash=seven.digest(report['selection']),image_hash=seven.digest(report['image_sha256']),
        model=seven.MODEL,rules=report['rules'],conditions=list(CONDITIONS),rotation=rotation,feature='final answer-prefix token, all 32 decoder-block outputs',
        seed=SEED,smoke=smoke,source_hashes={name:seven.file_hash(source_path(name)) for name in FEATURE_SOURCES})
    with seven.output(out,cfg,resume):
        seven.atomic_json(out/'input_report.json',report)
        jobs=[];records=[];dimension=None
        for item in work:
            for n in CONDITIONS:
                selected=report['selection'][item['set_id']][str(n)]
                expected=dict(set_id=item['set_id'],split=item['split'],images=n,true_rule=item['rule'],
                    image_files=selected['image_files'],rotation=rotation,selection_hash=seven.digest(selected))
                folder=feature_dir(out,item['set_id'],n)
                if folder.exists():
                    row,value=feature_record(folder,expected,helpers)
                    if dimension is not None and value.shape[1]!=dimension:raise ValueError('Hidden dimensions disagree')
                    dimension=value.shape[1];records.append(row)
                else:jobs.append((item,n,selected,expected,folder))
        mirror=out/'model_predictions.jsonl'
        if mirror.exists():
            text=mirror.read_text()
            if text and not text.endswith('\n'):raise ValueError('Partial prediction mirror; inspect before resuming')
            old=[json.loads(line) for line in text.splitlines()]
            keys=[(r['set_id'],r['images']) for r in old]
            if len(keys)!=len(set(keys)):raise ValueError('Duplicate prediction mirror records')
            expected_keys={(s['set_id'],n) for s in work for n in CONDITIONS}
            if not set(keys)<=expected_keys:raise ValueError('Unexpected prediction mirror record')
            indexed={(r['set_id'],r['images']):r for r in records}
            if any(key not in indexed or row!=indexed[key] for key,row in zip(keys,old)):raise ValueError('Prediction mirror disagrees with committed feature records')
        print(f'EXTRACTION: {len(work)} sets, {len(work)*3} planned forwards, {len(jobs)} pending. GPU model loads only if pending.',flush=True)
        if jobs:
            runtime={p:importlib.metadata.version(p) for p in INFERENCE_PACKAGES}
            if runtime!=INFERENCE_PACKAGES:raise ValueError('Match the pinned seven-rule inference environment before extracting features')
            if (out/'extraction_runtime.json').exists() and seven.read(out/'extraction_runtime.json')!=runtime:
                raise ValueError('Inference runtime differs from existing extracted features')
            seven.atomic_json(out/'extraction_runtime.json',runtime)
            model,processor,_=choices.shared.load_model();tokens=choices.label_tokens(processor.tokenizer)
            for item,n,selected,expected,folder in jobs:
                text=prompt(report['rules'],n,rotation)
                values,probs,count=helpers.capture(model,processor,tokens,selected['image_files'],data,text,seven.MODEL['max_pixels'])
                if values.shape[0]!=32 or (dimension is not None and values.shape[1]!=dimension):raise ValueError('Unexpected decoder shape')
                dimension=values.shape[1]
                mapping=choices.mapping(7,rotation);label=max(seven.LABELS,key=probs.get)
                row=dict(expected,probabilities=probs,predicted_label=label,predicted_rule=mapping[label],
                    correct=mapping[label]==item['rule'],input_tokens=count,ambiguous=(n==1))
                # Commit features and their prediction metadata as one directory.
                folder.parent.mkdir(parents=True,exist_ok=True);temp=folder.with_name(folder.name+'.tmp')
                if temp.exists():shutil.rmtree(temp)
                temp.mkdir();helpers.save_features(temp/'features.npz',values,count)
                row['feature_sha256']=seven.file_hash(temp/'features.npz');seven.atomic_json(temp/'metadata.json',row)
                temp.rename(folder);records.append(row)
                feature_record(folder,expected,helpers)
                print(f"{len(records)}/{len(work)*3} {item['set_id']} n={n} shape={values.shape} predicted={mapping[label]} ambiguous={n==1}",flush=True)
            del model,processor
        ordered=sorted(records,key=lambda r:(r['set_id'],r['images']))
        temp=out/'model_predictions.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in ordered));temp.replace(out/'model_predictions.jsonl')
        seven.atomic_json(out/'extraction_complete.json',dict(feature_jobs=len(ordered),smoke=smoke,source_sets_hash=report['source_sets_hash']))


def fit_layer(x_train,y_train,x_val,seed=SEED,shuffled=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    labels=np.array(y_train).copy()
    if shuffled:np.random.default_rng(seed).shuffle(labels)
    estimator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=SEED))
    estimator.fit(x_train,labels)  # No validation labels/statistics enter fitting.
    return estimator,estimator.predict(x_val),estimator.predict_proba(x_val)


def fit(out,items,report,rotation):
    out=Path(out);helpers=capture_helpers();cfg=seven.read(out/'config.json')
    current={name:seven.file_hash(source_path(name)) for name in FEATURE_SOURCES}
    if cfg.get('source_hashes')!=current:
        raise ValueError('Extraction source changed; fit or resume with its original commit')
    if cfg['smoke'] or cfg['source_sets_hash']!=report['source_sets_hash'] or cfg['selection_hash']!=seven.digest(report['selection']) or cfg['rotation']!=rotation or cfg['source_manifest']!=report['source_manifest_canonical_hash'] or cfg['image_hash']!=seven.digest(report['image_sha256']):
        raise ValueError('Need complete matching non-smoke extraction')
    train=[s for s in items if s['split']=='train'];val=[s for s in items if s['split']=='validation']
    if len(train)!=147 or len(val)!=49:raise ValueError('Original train/validation split changed')
    expected_keys={(s['set_id'],n) for s in items for n in CONDITIONS}
    log=[json.loads(line) for line in (out/'model_predictions.jsonl').read_text().splitlines()]
    if len(log)!=588 or {(r['set_id'],r['images']) for r in log}!=expected_keys:raise ValueError('Missing/duplicate extraction records')
    indexed={(r['set_id'],r['images']):r for r in log}
    arrays={};width=None
    for item in items:
        for n in CONDITIONS:
            selected=report['selection'][item['set_id']][str(n)]
            expected=dict(set_id=item['set_id'],split=item['split'],images=n,true_rule=item['rule'],image_files=selected['image_files'],rotation=rotation,selection_hash=seven.digest(selected))
            row,value=feature_record(feature_dir(out,item['set_id'],n),expected,helpers)
            if row!=indexed[(item['set_id'],n)]:raise ValueError('Prediction mirror disagrees with committed features')
            if width is not None and value.shape[1]!=width:raise ValueError('Feature dimensions disagree')
            width=value.shape[1];arrays[(item['set_id'],n)]=value
    accuracies=[];predictions=[];matrices=[];per_rule=[]
    y_train=np.array([s['rule'] for s in train]);y_val=np.array([s['rule'] for s in val])
    for n in CONDITIONS:
        tx=np.stack([arrays[(s['set_id'],n)] for s in train]);vx=np.stack([arrays[(s['set_id'],n)] for s in val])
        for layer in range(32):
            for shuffled in (False,True):
                estimator,pred,probs=fit_layer(tx[:,layer],y_train,vx[:,layer],SEED+layer,shuffled)
                control='shuffled_training_labels' if shuffled else 'true_training_labels'
                acc=float(np.mean(pred==y_val));norm=estimator.named_steps['standardscaler']
                accuracies.append(dict(images=n,layer=layer+1,control=control,accuracy=acc,train_sets=147,validation_sets=49,normalizer_fit_samples=int(norm.n_samples_seen_)))
                for rule in choices.IDS:
                    mask=y_val==rule;per_rule.append(dict(images=n,layer=layer+1,control=control,true_rule=rule,accuracy=float(np.mean(pred[mask]==rule))))
                    for target in choices.IDS:matrices.append(dict(images=n,layer=layer+1,control=control,true_rule=rule,predicted_rule=target,count=int(np.sum((y_val==rule)&(pred==target)))))
                for i,item in enumerate(val):predictions.append(dict(images=n,layer=layer+1,control=control,set_id=item['set_id'],true_rule=item['rule'],predicted_rule=str(pred[i]),correct=int(pred[i]==item['rule']),true_probability=float(probs[i,list(estimator.classes_).index(item['rule'])])))
                print(f'CPU probe n={n} layer={layer+1:02d} {control}: {acc:.2%} validation accuracy',flush=True)
    fit_out=out/'probes';fit_out.mkdir(exist_ok=True)
    seven.atomic_json(fit_out/'config.json',dict(protocol='seven-rule-linear-probes-v1',model=cfg['model'],rules=report['rules'],extraction_config_hash=seven.digest(cfg),train_sets=147,validation_sets=49,probe='train-only StandardScaler + LogisticRegression C=1 max_iter=2000',seed=SEED))
    for name,rows in [('layer_accuracy.csv',accuracies),('probe_predictions.csv',predictions),('confusion_matrices.csv',matrices),('per_rule_accuracy.csv',per_rule)]:
        with (fit_out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='CPU probes fitted to extracted frozen features',train_sets=147,validation_sets=49,layers=32,conditions=list(CONDITIONS),chance=1/7,
        fixed_rotation=rotation,source_manifest=report['source_manifest_canonical_hash'],extraction_config_hash=seven.digest(cfg),
        final_layer_predeclared=[r for r in accuracies if r['layer']==32],
        best_layer_selection='No best layer is selected. All 32 curves are descriptive validation results, not independent layer-selection test scores.',
        methodology=report['limitation'],shared_images='Images repeat within each split; sets are not independent. Source-image partitions are disjoint.',
        feature_hashes={f'{sid}:{n}':seven.file_hash(feature_dir(out,sid,n)/'features.npz') for sid,n in sorted(expected_keys)},
        software={p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','matplotlib')})
    seven.atomic_json(fit_out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for n in CONDITIONS:
        for control in ('true_training_labels','shuffled_training_labels'):
            rows=[r for r in accuracies if r['images']==n and r['control']==control]
            ax.plot([r['layer'] for r in rows],[r['accuracy'] for r in rows],linestyle='--' if control.startswith('shuffled') else '-',label=f'{n} images / {control}')
    ax.axhline(1/7,color='gray',linestyle=':');ax.set(xlabel='Decoder layer',ylabel='Validation accuracy',ylim=(0,1));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(fit_out/'layer_accuracy.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for col,n in enumerate(CONDITIONS):
        for row,control in enumerate(('true_training_labels','shuffled_training_labels')):
            matrix=[[next(r['count'] for r in matrices if r['images']==n and r['layer']==32 and r['control']==control and r['true_rule']==truth and r['predicted_rule']==pred) for pred in choices.IDS] for truth in choices.IDS]
            axes[row,col].imshow(matrix);axes[row,col].set(title=f'{n} images / {control}',xticks=range(7),yticks=range(7),xticklabels=choices.IDS,yticklabels=choices.IDS,xlabel='Predicted',ylabel='True')
    fig.tight_layout();fig.savefig(fit_out/'final_layer_confusions.png');plt.close(fig)
    (fit_out/'summary.md').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='feature_hashes'},indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--validate-only',action='store_true');mode.add_argument('--smoke-test',action='store_true');mode.add_argument('--fit-only',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--rotation',type=int,choices=range(7),default=0)
    a=p.parse_args();items,report=plan_inputs(a.manifest,a.data_dir)
    print(json.dumps(dict(splits=report['split_statistics'],minimum_configurations=report['minima'],full_extraction_forwards=588,smoke_forwards=3,conditions=[1,5,10],rotation=a.rotation,methodology=report['limitation']),indent=2),flush=True)
    if a.validate_only:return
    out=validation.local_path(a.out)
    if a.fit_only:fit(out,items,report,a.rotation);return
    extract(out,a.data_dir,items,report,a.rotation,a.resume,a.smoke_test)
    if not a.smoke_test:fit(out,items,report,a.rotation)


if __name__=='__main__':main()
