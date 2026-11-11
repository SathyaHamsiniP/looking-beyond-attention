"""Experiment 08: bounded seven-rule validation patching using historical hooks."""
import argparse
from collections import defaultdict, OrderedDict
import csv
import importlib.util
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import REPO, source_path

from types import SimpleNamespace

import seven_rule_validation as validation
from seven_rule_validation import seven,choices


HISTORICAL=REPO/'three-operand-experiments/code/03-activation-patching/patch_activations_qwen35.py'


def intervention_engine(tokens):
    # Reuse the historical capture and replacement hooks unchanged. Only the
    # isolated module's readout/rule vocabulary is adapted from XYZ to 1..7.
    spec=importlib.util.spec_from_file_location('seven_rule_historical_hooks',HISTORICAL)
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    def probabilities(model,inputs,mapping):
        import torch
        with torch.inference_mode():
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            scores=logits[tokens]
            if not torch.isfinite(scores).all():raise ValueError('Nonfinite patch readout')
            values=scores.softmax(0).cpu().tolist()
        return {mapping[label]:float(p) for label,p in zip(seven.LABELS,values)}
    engine.probabilities=probabilities;engine.RULES=tuple(choices.IDS)
    return engine


def make_pairs(items,pairs_per_rule=1,offsets=(1,)):
    if not 1<=pairs_per_rule<=7 or not offsets or len(set(offsets))!=len(offsets) or any(o not in range(1,7) for o in offsets):raise ValueError('Use 1..7 receivers/rule and distinct donor offsets 1..6')
    by_rule={r:sorted([s for s in items if s['rule']==r],key=lambda s:s['set_id']) for r in choices.IDS}
    if any(len(v)!=7 for v in by_rule.values()):raise ValueError('Need seven validation sets per rule')
    pairs=[]
    for ri,rule in enumerate(choices.IDS):
        for index in range(pairs_per_rule):
            receiver=by_rule[rule][index]
            for offset in offsets:
                group=f'{rule}:{index}:offset-{offset}'
                for condition,donor in [('same',by_rule[rule][(index+offset)%7]),
                    ('different',by_rule[choices.IDS[(ri+offset)%7]][(index+offset)%7])]:
                    if receiver['set_id']==donor['set_id']:raise ValueError('Self donor')
                    pairs.append(dict(pair_id=group+':'+condition,control_group=group,condition=condition,
                        receiver_set=receiver['set_id'],receiver_rule=rule,donor_set=donor['set_id'],donor_rule=donor['rule'],
                        shared_images=len(set(receiver['image_files'])&set(donor['image_files']))))
    return pairs


def effect(pair,rotation,layer,position,before,donor,after):
    for probabilities in (before,donor,after):
        if set(probabilities)!=set(choices.IDS) or not all(math.isfinite(v) and 0<=v<=1 for v in probabilities.values()) or not math.isclose(sum(probabilities.values()),1,abs_tol=1e-6):raise ValueError('Invalid seven-rule distribution')
    mapping=choices.mapping(7,rotation)
    donor_rule=pair['donor_rule'];receiver_prediction=max(before,key=before.get)
    donor_prediction=max(donor,key=donor.get);patched=max(after,key=after.get)
    donor_label=next(l for l,r in mapping.items() if r==donor_prediction)
    label_target=mapping[donor_label]
    delta=after[donor_rule]-before[donor_rule]
    return dict(pair,rotation=rotation,layer=layer,position=position,donor_true_rule=donor_rule,
        donor_predicted_rule=donor_prediction,donor_original_predicted_label=donor_label,
        receiver_original_predicted_rule=receiver_prediction,patched_predicted_rule=patched,
        delta_true_rule=delta,absolute_delta_true_rule=abs(delta),
        delta_donor_answer=after[label_target]-before[label_target],
        total_variation=.5*sum(abs(after[r]-before[r]) for r in choices.IDS),
        top_rule_changed=int(patched!=receiver_prediction),
        before_probabilities=before,donor_probabilities=donor,patched_probabilities=after)


def summarize(rows):
    clusters=defaultdict(lambda:defaultdict(list))
    fields=('delta_true_rule','absolute_delta_true_rule','delta_donor_answer','total_variation','top_rule_changed')
    for row in rows:clusters[(row['condition'],row['position'],row['layer'])][row['pair_id']].append(row)
    summaries=[]
    for (condition,position,layer),pairs in sorted(clusters.items()):
        entry=dict(condition=condition,position=position,layer=layer,pair_clusters=len(pairs))
        for field in fields:
            values=[sum(r[field] for r in group)/len(group) for group in pairs.values()]
            entry[field]=sum(values)/len(values)
            entry[field+'_95ci_pair']=validation.bootstrap(values)
        summaries.append(entry)
    controls=defaultdict(dict)
    for row in rows:controls[(row['control_group'],row['position'],row['layer'],row['rotation'])][row['condition']]=row
    differences=defaultdict(lambda:defaultdict(list))
    for (group,pos,layer,rotation),conditions in controls.items():
        if set(conditions)=={'same','different'}:
            differences[(pos,layer)][group].append(conditions['different']['delta_true_rule']-conditions['same']['delta_true_rule'])
    comparisons=[]
    for (pos,layer),groups in sorted(differences.items()):
        values=[sum(v)/len(v) for v in groups.values()]
        comparisons.append(dict(position=pos,layer=layer,different_minus_same=sum(values)/len(values),
            ci95_matched_pair_groups=validation.bootstrap(values)))
    late=[]
    for pos in sorted({r['position'] for r in rows}):
        groups=defaultdict(lambda:defaultdict(list))
        for (p,layer),values in differences.items():
            if p==pos:
                for group,v in values.items():groups[group]['late' if layer>=25 else 'early' if layer<=8 else 'middle'].append(sum(v)/len(v))
        vals=[sum(v['late'])/len(v['late'])-sum(v['early'])/len(v['early']) for v in groups.values() if v['late'] and v['early']]
        if vals:late.append(dict(position=pos,late_25_32_minus_early_1_8_control_adjusted=sum(vals)/len(vals),ci95=validation.bootstrap(vals)))
    return dict(layer_effects=summaries,matched_controls=comparisons,late_layer_comparison=late,
        sign='patched minus original receiver probability',
        uncertainty='5000 percentile bootstrap draws of donor-receiver pairs/matched control groups, seed 2026; shared sets/images across pairs remain dependent',
        caution='True-rule transfer, donor-predicted numeric-answer transfer, and distribution perturbation are distinct. Negative shifts do not disprove rule representations. Answer copying does not prove reasoning. Position bias, donor errors, shared images and intervention artifacts confound interpretation.')


def save_analysis(out,rows):
    report=summarize(rows);seven.atomic_json(out/'summary.json',report)
    simple=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
    with (out/'patches.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(simple[0]));writer.writeheader();writer.writerows(simple)
    with (out/'layer_effects.csv').open('w',newline='') as f:
        cols=['condition','position','layer','pair_clusters','delta_true_rule','absolute_delta_true_rule','delta_donor_answer','total_variation','top_rule_changed']
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(report['layer_effects'])
    for filename,keys in [('pair_effects.csv',('pair_id','condition','position','layer')),('per_rule_effects.csv',('donor_true_rule','receiver_rule','condition','position','layer'))]:
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        aggregated=[]
        for identity,group in groups.items():
            values=dict(zip(keys,identity));values['repeated_measurements']=len(group)
            for field in ('delta_true_rule','absolute_delta_true_rule','delta_donor_answer','total_variation','top_rule_changed'):
                values[field]=sum(r[field] for r in group)/len(group)
            aggregated.append(values)
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(aggregated[0]));writer.writeheader();writer.writerows(aggregated)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for condition in ('same','different'):
        for position in sorted({r['position'] for r in rows}):
            group=[r for r in report['layer_effects'] if r['condition']==condition and r['position']==position]
            for ax,field in zip(axes,('delta_true_rule','delta_donor_answer','total_variation')):
                ax.plot([r['layer'] for r in group],[r[field] for r in group],label=f'{condition}/{position}');ax.set(xlabel='Decoder layer output',ylabel=field)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'layer_effects.png');plt.close(fig)
    print('condition position layer true-rule-delta answer-delta total-variation',flush=True)
    for r in report['layer_effects']:print(r['condition'],r['position'],r['layer'],round(r['delta_true_rule'],6),round(r['delta_donor_answer'],6),round(r['total_variation'],6),flush=True)
    print(json.dumps({'late_layer_comparison':report['late_layer_comparison'],'caution':report['caution']},indent=2),flush=True)
    return report


def run(view,data,cache,out,pairs,rotations,layers,positions,resume=False,training_root=None,manifest_path=None):
    # Every result row is keyed by a fixed plan, and only pending interventions run.
    cached,recovered,saved_tokens=validation.verify_baseline(cache,view,config_name='recovered_config.json')
    if seven.read(Path(cache)/'validation_view.json')!=view:raise ValueError('A3 cache validation inputs differ')
    adapter=None;selected_identity=None
    if training_root:
        training_root=Path(training_root)
        if not (training_root/'training/complete.json').is_file():
            print('Optional selected-model comparison skipped: completed seven-rule adapter is unavailable.');return
        if manifest_path is None:raise ValueError('Full source manifest is required for selected-adapter identity verification')
        original=seven.read(manifest_path)
        if seven.digest(original)!=view['source_manifest_canonical_hash']:raise ValueError('Adapter source-manifest identity differs')
        seven.require_selection(training_root,original)
        adapter=training_root/'training/best_adapter';selected_identity=seven.adapter_identity(adapter)
    expected={f"{p['pair_id']}:{r}:{l}:{pos}":(p,r,l,pos) for p in pairs for r in rotations for l in layers for pos in positions}
    config=dict(view_hash=seven.digest(view),pairs=pairs,rotations=rotations,layers=layers,positions=positions,
        intervention='historical decoder-block output replacement at final/previous token',model=seven.MODEL,
        selected_adapter_identity=selected_identity,model_variant='selected seven-rule adapter' if adapter else 'pretrained',
        source_hashes={name:seven.file_hash(source_path(name)) for name in ('benchmark_paths.py','activation_patching_7rules.py','seven_rule_validation.py','lora_7rules.py','compare_3_vs_7.py')},
        historical_hook_hash=seven.file_hash(HISTORICAL),baseline_hash=seven.file_hash(Path(cache)/'predictions.jsonl'))
    with seven.output(out,config,resume):
        saved={};path=out/'patches.jsonl'
        if path.exists():
            content=path.read_text()
            if content and not content.endswith('\n'):raise ValueError('Partial patch log line; inspect before resuming')
            for line in content.splitlines():
                row=json.loads(line);key=row['key']
                if key in saved or key not in expected:raise ValueError('Duplicate/unexpected patch row')
                p,r,l,pos=expected[key]
                recomputed=effect(p,r,l,pos,row['before_probabilities'],row['donor_probabilities'],row['patched_probabilities'])
                if row!=dict(recomputed,key=key):raise ValueError('Corrupt saved patch effects')
                saved[key]=row
        pending={key:value for key,value in expected.items() if key not in saved}
        donors={(p['donor_set'],r) for p,r,_,_ in pending.values()}
        receiver_extras={(p['receiver_set'],r) for p,r,_,_ in pending.values()}-donors if adapter else set()
        workload=dict(pending_interventions=len(pending),donor_capture_forwards=len(donors),receiver_baseline_forwards=len(receiver_extras),total_forward_upper_bound=len(pending)+len(donors)+len(receiver_extras))
        print('GPU WORKLOAD:',json.dumps(workload),flush=True);seven.atomic_json(out/'workload.json',workload)
        if pending:
            import torch
            import importlib.metadata
            for package in ('torch','transformers','bitsandbytes','accelerate','pillow'):
                if importlib.metadata.version(package)!=recovered['software'][package]:raise ValueError(f'{package} differs from recovered inference environment; match pinned dependencies before patching')
            model,processor,_=choices.shared.load_model()
            if adapter:
                from peft import PeftModel
                model=PeftModel.from_pretrained(model,adapter,is_trainable=False).eval()
            tokens=choices.label_tokens(processor.tokenizer)
            if (Path(cache)/'answer_tokens.json').exists() and dict(zip(seven.LABELS,tokens))!=seven.read(Path(cache)/'answer_tokens.json'):raise ValueError('Recovered tokenizer IDs differ')
            engine=intervention_engine(tokens);blocks=engine.decoder_layers(model.get_base_model() if adapter else model)
            if len(blocks)!=32:raise ValueError('Expected 32 decoder layers')
            wrapper=SimpleNamespace(model=model);by_id={s['set_id']:s for s in view['sets']};vectors={};inputs=OrderedDict()
            def prepared(sid,rotation):
                key=(sid,rotation)
                if key in inputs:inputs.move_to_end(key);return inputs[key]
                item=by_id[sid];prompt=choices.prompt_for(view['rules'],choices.mapping(7,rotation))
                value=seven.prepare(model,processor,item,data,prompt,tokens);inputs[key]=value
                if len(inputs)>2:inputs.popitem(last=False)
                return value
            selected_originals={}
            original_log=out/'selected_unpatched.jsonl'
            if adapter and original_log.exists():
                for line in original_log.read_text().splitlines():
                    item=json.loads(line);original_key=(item['set_id'],item['rotation'])
                    if original_key in selected_originals:raise ValueError('Duplicate selected baseline')
                    selected_originals[original_key]=item['probabilities']
            def capture(key,mapping):
                print(f'Capturing donor/receiver vectors once: {key}',flush=True)
                value=engine.record_original(wrapper,prepared(*key),mapping,blocks)
                if adapter:
                    if key in selected_originals and max(abs(value['probabilities'][r]-selected_originals[key][r]) for r in choices.IDS)>1e-5:raise ValueError('Selected-model baseline drift on resume')
                    if key not in selected_originals:
                        seven.append(original_log,dict(set_id=key[0],rotation=key[1],probabilities=value['probabilities']))
                        selected_originals[key]=value['probabilities']
                return value
            for key,(pair,rotation,layer,pos) in pending.items():
                donor_key=(pair['donor_set'],rotation);mapping=choices.mapping(7,rotation)
                if donor_key not in vectors:
                    captured=capture(donor_key,mapping)
                    baseline=cached[f'{donor_key[0]}:{rotation}']
                    expected_probs={mapping[label]:p for label,p in baseline['probabilities'].items()}
                    if not adapter and max(abs(captured['probabilities'][r]-expected_probs[r]) for r in choices.IDS)>1e-5:
                        raise ValueError('Fresh donor readout disagrees with recovered baseline; stop instead of mixing incompatible runs')
                    vectors[donor_key]=captured
                receiver=cached[f"{pair['receiver_set']}:{rotation}"]
                if adapter:
                    receiver_key=(pair['receiver_set'],rotation)
                    if receiver_key not in vectors:vectors[receiver_key]=capture(receiver_key,mapping)
                    before=vectors[receiver_key]['probabilities']
                else:before={mapping[label]:p for label,p in receiver['probabilities'].items()}
                donor=vectors[donor_key]
                patched=engine.patched_probabilities(wrapper,prepared(pair['receiver_set'],rotation),mapping,
                    blocks[layer-1],donor['activations'][layer][pos],pos)
                if adapter:donor_reference=donor['probabilities']
                else:
                    original_donor=cached[f"{pair['donor_set']}:{rotation}"]
                    donor_reference={mapping[label]:p for label,p in original_donor['probabilities'].items()}
                    if donor['prediction']!=original_donor['predicted_rule']:raise ValueError('Fresh donor prediction differs from recovered original')
                row=dict(effect(pair,rotation,layer,pos,before,donor_reference,patched),key=key)
                seven.append(path,row);saved[key]=row
                print(f"Patching progress {len(saved)}/{len(expected)}; {pair['condition']} L{layer}/{pos}: true-rule delta={row['delta_true_rule']:+.6f}",flush=True)
            inputs.clear();vectors.clear();del model,processor;torch.cuda.empty_cache()
        return save_analysis(out,list(saved.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pairs-per-rule',type=int,default=1);p.add_argument('--donor-offsets',nargs='+',type=int,default=[1])
    p.add_argument('--rotations',nargs='+',type=int,default=[0]);p.add_argument('--layers',nargs='+',type=int,default=list(range(1,33)))
    p.add_argument('--positions',nargs='+',choices=('final','previous'),default=['final','previous'])
    p.add_argument('--smoke',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--optional-training-root',type=Path)
    a=p.parse_args();view=validation.validation_view(a.manifest,a.data_dir)
    for values,allowed in ((a.rotations,set(range(7))),(a.layers,set(range(1,33))),(a.positions,{'final','previous'})):
        if not values or len(set(values))!=len(values) or not set(values)<=allowed:p.error('Invalid/duplicate rotations, layers or positions')
    pairs=make_pairs(view['sets'],a.pairs_per_rule,a.donor_offsets)
    if a.smoke:pairs=pairs[:2];a.rotations=[0];a.layers=[1,32];a.positions=['final','previous']
    donors={(p['donor_set'],r) for p in pairs for r in a.rotations}
    n=len(pairs)*len(a.rotations)*len(a.layers)*len(a.positions)
    print(f'PLAN: {len(pairs)} pairs, {n} interventions + at most {len(donors)} donor forwards = {n+len(donors)} forwards; receiver baselines reused. Smoke={a.smoke}',flush=True)
    if not a.run:
        print('CPU-only plan. Add --run explicitly after checking the workload.');return
    run(view,a.data_dir,a.cache,validation.local_path(a.out),pairs,a.rotations,a.layers,a.positions,a.resume,a.optional_training_root,a.manifest)


if __name__=='__main__':main()
