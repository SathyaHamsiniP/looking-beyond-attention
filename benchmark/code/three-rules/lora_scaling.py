"""Controlled 24/45/90-set QLoRA scaling, using pinned historical primitives.

No historical source is copied or modified. --source-repo is a separate clean,
detached checkout of SOURCE_COMMIT. Default evaluation is validation only.
The historical 90-set run is archival, not the matched full-batch condition.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00-common'))
from benchmark_paths import REPO, source_path

import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

SOURCE_COMMIT = '680e8a7d51ac36b08c60e11cc0ee7280404ee65d'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MANIFEST_HASH = '701936daac921fd19da203fccda2524d43c7277999ebdee0412041d987f1fcd4'
BUDGETS = (24, 45, 90)
RULES = ('R1', 'R2', 'R3')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def source(repo, gpu=False):
    repo = Path(repo).resolve()
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != SOURCE_COMMIT:
        raise ValueError(f'Historical helper checkout must be exactly {SOURCE_COMMIT}; found {sha}')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Historical helper checkout has tracked modifications')
    folder = repo/'three-operand-experiments/code/04-lora'
    if gpu:
        sys.path.insert(0, str(folder))
        import train_lora
        import common
        if Path(train_lora.__file__).resolve() != folder/'train_lora.py' or Path(common.__file__).resolve() != folder/'common.py':
            raise ValueError('Conflicting historical module imports; use a fresh Python process')
        return train_lora, common
    spec = importlib.util.spec_from_file_location('scaling_original_sets', folder/'build_training_sets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_subsets(manifest, seed=2026):
    train = [s for s in manifest['sets'] if s['partition'] == 'train']
    pools = {}
    for rule in RULES:
        if sum(s['rule']==rule for s in train)!=30:
            raise ValueError('Expected 30 original training sets per rule')
        for label in 'XYZ':
            pool=sorted(s['set_id'] for s in train if s['rule']==rule and
                        next(l for l,r in s['option_mapping'].items() if r==rule)==label)
            if len(pool)!=10:
                raise ValueError('Original training data must have ten sets per rule/answer-label stratum')
            random.Random(f'{seed}:{rule}:{label}').shuffle(pool)
            pools[(rule,label)]=pool
    result = {}
    for n in BUDGETS:
        chosen=set()
        per_rule=n//3
        for ri,rule in enumerate(RULES):
            # For 24: 3/3/2 per rule, rotating the short label, totaling 8/8/8.
            # For 45 and 90: 5 and 10 per rule/label stratum, respectively.
            order='XYZ'[ri:]+'XYZ'[:ri]
            for li,label in enumerate(order):
                count=per_rule//3+int(li<per_rule%3)
                chosen.update(pools[(rule,label)][:count])
        result[n] = [s for s in train if s['set_id'] in chosen]
    return result


def schedule(items, steps, seed):
    """Continuous shuffled passes; every optimizer update contains four sets."""
    stream = []
    epoch = 0
    while len(stream) < steps*4:
        order = list(range(len(items)))
        random.Random(seed+epoch).shuffle(order)
        stream.extend(items[i]['set_id'] for i in order)
        epoch += 1
    return [stream[i:i+4] for i in range(0, steps*4, 4)]


def describe(items):
    counts = Counter(f for s in items for f in s['image_files'])
    return dict(sets=len(items), sets_per_rule=dict(Counter(s['rule'] for s in items)),
        answer_label_counts=dict(Counter(next(l for l,r in s['option_mapping'].items() if r==s['rule']) for s in items)),
        set_ids=[s['set_id'] for s in items], selected_sets_hash=digest(items),
        unique_source_images=len(counts), image_appearances=sum(counts.values()),
        images_reused_across_sets=sum(v>1 for v in counts.values()),
        image_reuse_counts=dict(sorted(counts.items())))


def make_plan(manifest, steps=46, seed=2026, subset_seed=2026):
    if steps < 1:
        raise ValueError('Optimizer-step budget must be positive')
    if digest(manifest) != MANIFEST_HASH:
        raise ValueError('Use the exact completed three-rule Experiment 04 manifest; no regenerated manifest')
    if len({s['set_id'] for s in manifest['sets']}) != len(manifest['sets']):
        raise ValueError('Duplicate original set IDs')
    by_split = {split: [s for s in manifest['sets'] if s['partition']==split] for split in ('train','validation','test')}
    used = {split: {f for s in items for f in s['image_files']} for split, items in by_split.items()}
    if any(used[a]&used[b] for a in used for b in used if a != b):
        raise ValueError('Cross-partition source-image leakage')
    subsets = select_subsets(manifest, subset_seed)
    for a, b in zip(BUDGETS, BUDGETS[1:]):
        if not {s['set_id'] for s in subsets[a]} < {s['set_id'] for s in subsets[b]}:
            raise ValueError('Training subsets are not strictly nested')
    common = dict(model='Qwen/Qwen3.5-4B', revision=REVISION, precision='4bit',
        quantization='NF4, double quantization, FP16 compute', rank=8, alpha=16,
        dropout=.05, optimizer='torch.optim.AdamW', learning_rate=.0002,
        optimizer_defaults=dict(betas=[.9,.999], eps=1e-8, weight_decay=.01),
        batch_size=1, gradient_accumulation=4, initial_gradient_scale=8.,
        max_pixels=100352, max_tokens=16000, seed=seed, successful_optimizer_steps=steps,
        validation_frequency=10, checkpoint_frequency=1, gradient_clip=1.,
        initialization='fresh frozen pretrained base and default PEFT LoRA initialization for each condition; same seed',
        target_modules='all decoder linear modules resolved by historical targets(); vision frozen',
        training_options='original saved option_mapping; unchanged between revisits',
        scoring='historical conditional answer-sequence likelihood normalized over XYZ',
        selection='lowest full-vocabulary answer-token loss over all validation sets and three rotations',
        final_test_used=False)
    conditions = {}
    for n, items in subsets.items():
        conditions[str(n)] = dict(describe(items), actual_fraction=n/90,
            effective_passes=steps*4/n, set_presentations=steps*4,
            update_schedule=schedule(items, steps, seed), training_config=common.copy())
    return dict(protocol='three-rule-qlora-set-scaling-v1', source_commit=SOURCE_COMMIT,
        manifest_hash=digest(manifest), subset_seed=subset_seed, conditions=conditions,
        validation=describe(by_split['validation']), test=describe(by_split['test']),
        previous_100_percent=dict(directly_comparable=False, role='archival reference only',
            reason='Historical run: 180 set presentations, partial batches at steps 23/46, 46 updates. New protocol: 184 presentations in 46 full four-set updates, continuous shuffled passes; checkpoint every update.',
            recorded_updates=46, selected_step=20, best_validation_loss=1.1470619327492184,
            validation_accuracy=30/90, all_validation_answers='Y'),
        limitations='Single seed initially; shared images within partitions; positive-only decisive-image shortcuts; accuracy does not establish logical reasoning. Historical test partition was reserved by this LoRA run, but other exploratory analyses mean global untouched status is not established.')


def output_path(root, seed, condition, smoke=False):
    if condition not in ('pretrained', *map(str, BUDGETS)):
        raise ValueError('Unknown condition')
    return Path(root)/f'seed-{seed}'/('smoke' if smoke else 'matched')/condition


@contextmanager
def locked_output(path, config, resume):
    import fcntl
    path = Path(path)
    if path.exists() and not resume:
        raise ValueError(f'Output exists: {path}; explicitly --resume or choose a new root')
    if resume and not path.exists():
        raise ValueError('Cannot resume a nonexistent output directory')
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process is using this condition') from exc
        if resume:
            if read(path/'config.json') != config:
                raise ValueError('Resume configuration/manifest/source/software differs')
        else:
            write(path/'config.json', config)
        yield path


def check_prediction(row):
    probs=row['probabilities']
    if set(probs)!=set(RULES) or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in probs.values()) or not math.isclose(sum(probs.values()),1.,abs_tol=1e-6):
        raise ValueError('Invalid saved prediction probabilities')
    prediction=max(probs,key=probs.get)
    if row['prediction']!=prediction or row['mapping'][row['answer_label']]!=prediction or type(row['correct']) is not bool or row['correct']!=(prediction==row['true_rule']):
        raise ValueError('Prediction, label, probabilities or correctness disagree')


def evaluate(adapter, common, items, data, path):
    expected = {f"{s['set_id']}:{r}" for s in items for r in range(3)}
    rows = common.rows(path/'predictions.jsonl')
    by_id = {s['set_id']:s for s in items}
    if not set(rows) <= expected:
        raise ValueError('Unexpected saved evaluation keys')
    for key, row in rows.items():
        check_prediction(row)
        item = by_id[row['set_id']]
        if row['image_files'] != item['image_files'] or row['mapping'] != common.mapping(row['rotation']) or row['true_rule'] != item['rule'] or key != f"{item['set_id']}:{row['rotation']}":
            raise ValueError('Saved evaluation inputs differ')
    adapter.model.eval()
    for item in items:
        for rotation in range(3):
            key = f"{item['set_id']}:{rotation}"
            if key in rows:
                continue
            probs = common.score(adapter, common.inputs(adapter,item,data,rotation), rotation)
            prediction = max(probs,key=probs.get)
            row = dict(set_id=item['set_id'],true_rule=item['rule'],image_files=item['image_files'],
                rotation=rotation,mapping=common.mapping(rotation),prediction=prediction,
                answer_label=next(l for l,r in common.mapping(rotation).items() if r==prediction),
                probabilities=probs,correct=prediction==item['rule'])
            check_prediction(row)
            common.append(path/'predictions.jsonl',key,row);rows[key]=row
    return list(rows.values())


def metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['set_id']].append(row)
    if not groups or any(sorted(r['rotation'] for r in g)!=[0,1,2] for g in groups.values()):
        raise ValueError('Metrics require all three rotations for every set')
    return dict(sets=len(groups), trials=len(rows), accuracy=sum(r['correct'] for r in rows)/len(rows),
        per_rule={rule:sum(r['correct'] for r in rows if r['true_rule']==rule)/sum(r['true_rule']==rule for r in rows) for rule in RULES},
        confusion={rule:{pred:sum(r['true_rule']==rule and r['prediction']==pred for r in rows) for pred in RULES} for rule in RULES},
        answer_labels={label:sum(r['answer_label']==label for r in rows) for label in 'XYZ'},
        rotation_consistency=sum(len({r['prediction'] for r in g})==1 for g in groups.values())/len(groups),
        unit='image set; rotations are repeated measurements; images may be shared across sets')


def checkpoint(path, adapter, state, common, meta):
    # Keep only latest and best, not one optimizer binary per update.
    temp, previous, latest = path/'latest.tmp', path/'latest.previous', path/'latest'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    adapter.model.save_pretrained(temp);adapter.processor.save_pretrained(temp)
    common.save(temp/'experiment.json',meta)
    import torch
    torch.save(state,temp/'training_state.pt')
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        latest.rename(previous)
    temp.rename(latest)
    if previous.exists():
        shutil.rmtree(previous)


def train(args, manifest, plan, config, path, trainer, common):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
    settings = config['training_config']
    torch.manual_seed(args.seed);random.seed(args.seed)
    params = SimpleNamespace(revision=REVISION, precision='4bit', max_pixels=100352, adapter=None)
    adapter = common.load_model(params, training=True)
    chosen = trainer.targets(adapter.model)
    if len(chosen)!=248:
        raise ValueError('Expected the original 248 decoder linear targets; model architecture or library resolution changed')
    for p in adapter.model.parameters():
        p.requires_grad=False
    adapter.model = prepare_model_for_kbit_training(adapter.model,use_gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={'use_reentrant':False})
    latest = path/'latest'
    if args.resume and not latest.exists() and (path/'latest.previous').exists():
        (path/'latest.previous').rename(latest)
    if latest.exists():
        adapter.model=PeftModel.from_pretrained(adapter.model,latest,is_trainable=True)
    else:
        adapter.model=get_peft_model(adapter.model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                        bias='none',task_type='CAUSAL_LM',target_modules=chosen))
    write(path/'trainable.json',trainer.trainable(adapter.model,chosen))
    optimizer=torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad],lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=8.)
    records=[];best=float('inf');start=0;selection=None
    if latest.exists():
        # Trusted, locally created optimizer/RNG state only.
        state=torch.load(latest/'training_state.pt',map_location='cpu',weights_only=False)
        if state['config_hash']!=digest(config):
            raise ValueError('Checkpoint configuration differs')
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        records=state['records'];best=state['best'];start=state['step'];selection=state['selection']
    by_id={s['set_id']:s for s in manifest['sets']}
    val=[s for s in manifest['sets'] if s['partition']=='validation']
    if args.smoke_test:
        val=[next(s for s in val if s['rule']==r) for r in RULES]
    total=2 if args.smoke_test else args.steps
    meta=dict(settings=common.settings(params),manifest_hash=MANIFEST_HASH,target_modules=chosen,
              scaling_config_hash=digest(config))
    def loss(item,rotation):
        prepared=common.inputs(adapter,item,args.data_dir,rotation)
        label=next(l for l,r in common.mapping(rotation).items() if r==item['rule'])
        ids=common.encode(adapter,label)
        if prepared['input_ids'].shape[1]+len(ids)>16000:
            raise ValueError('Input exceeds 16000 tokens; no image was dropped')
        return -common.answer_logp(adapter.model,prepared,ids)/len(ids)
    for index in range(start,total):
        adapter.model.train();optimizer.zero_grad(set_to_none=True);value=0.
        for sid in config['condition']['update_schedule'][index]:
            item=by_id[sid]
            rotation=next(r for r in range(3) if common.mapping(r)==item['option_mapping'])
            with torch.autocast('cuda',dtype=torch.float16):
                one=loss(item,rotation)
            if not torch.isfinite(one):
                raise ValueError('Non-finite training loss')
            value+=float(one.detach())/4;scaler.scale(one/4).backward()
        evidence=trainer.optimizer_update(adapter.model,optimizer,scaler)
        if not evidence['optimizer_step_occurred'] or not evidence['lora_b_weights_changed']:
            raise RuntimeError('Update did not succeed/change LoRA B; no successful-step credit. Resume from latest, inspect evidence: '+str(evidence))
        record=dict(step=index+1,training_loss=value,**evidence)
        if (index+1)%10==0 or index+1==total:
            adapter.model.eval()
            with torch.no_grad():
                record['validation_loss']=sum(float(loss(s,r)) for s in val for r in range(3))/(len(val)*3)
            if not math.isfinite(record['validation_loss']):
                raise ValueError('Non-finite validation loss')
            if record['validation_loss']<best:
                best=record['validation_loss']
                # Best adapter is also saved inside the transactional latest checkpoint
                # below, so interrupted selection updates can be recovered safely.
                selection=dict(step=index+1,loss=best,criterion=settings['selection'])
                best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(adapter.model).items()}
        records.append(record)
        state=dict(config_hash=digest(config),step=index+1,best=best,selection=selection,
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),records=records,
            best_weights=best_state if 'best_state' in locals() else state.get('best_weights') if latest.exists() else None)
        checkpoint(path,adapter,state,common,meta)
        (path/'losses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        print(record,flush=True)
    # The transactional checkpoint is authoritative after an interrupted log write.
    (path/'losses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    # Restore the selected adapter, not the final update, for evaluation/export.
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(adapter.model,state['best_weights'])
    adapter.model.save_pretrained(path/'best_adapter');adapter.processor.save_pretrained(path/'best_adapter')
    common.save(path/'best_adapter/experiment.json',meta)
    write(path/'selection.json',selection)
    write(path/'complete.json',dict(successful_optimizer_steps=total,selected_step=selection['step'],
        best_validation_loss=best,sets=config['condition']['sets'],set_presentations=total*4,
        effective_passes=total*4/config['condition']['sets'],test_used=False,smoke_test=args.smoke_test))


def compare_outputs(root, seed, plan):
    all_rows={};report={};fixed=None
    for condition in ('pretrained',*map(str,BUDGETS)):
        path=output_path(root,seed,condition)
        config=read(path/'config.json')
        if config['plan_hash']!=digest(plan):
            raise ValueError('Output uses a different scaling plan')
        current={k:config[k] for k in ('runner_sha256','software','model_settings','source_commit','manifest_hash','seed','smoke_test','partition')}
        if fixed is None:
            fixed=current
        elif current!=fixed:
            raise ValueError('Software, model, preprocessing, seed or runner differs across conditions')
        if condition!='pretrained' and config['training_config']!=plan['conditions'][condition]['training_config']:
            raise ValueError('Training configuration differs from the controlled plan')
        if condition!='pretrained':
            total=plan['conditions'][condition]['training_config']['successful_optimizer_steps']
            evidence=[json.loads(l) for l in (path/'losses.jsonl').read_text().splitlines()]
            if read(path/'complete.json')['successful_optimizer_steps']!=total or [r['step'] for r in evidence]!=list(range(1,total+1)) or not all(r['optimizer_step_occurred'] and r['lora_b_weights_changed'] and r['lora_b_matrices_changed']==248 for r in evidence):
                raise ValueError('Incomplete or unsuccessful optimizer-update evidence')
        rows=[json.loads(l) for l in (path/'predictions.jsonl').read_text().splitlines()]
        if {r['set_id'] for r in rows}!=set(plan['validation']['set_ids']):
            raise ValueError('Evaluation must use the unchanged validation sets')
        for row in rows:
            check_prediction(row)
        all_rows[condition]=rows;report[condition]=metrics(rows)
    target_lists=[read(output_path(root,seed,str(n))/'trainable.json')['target_modules'] for n in BUDGETS]
    if any(targets!=target_lists[0] for targets in target_lists[1:]):
        raise ValueError('Resolved LoRA targets differ between training conditions')
    base={(r['set_id'],r['rotation']):r for r in all_rows['pretrained']}
    for condition,rows in all_rows.items():
        for r in rows:
            b=base[(r['set_id'],r['rotation'])]
            if any(r[k]!=b[k] for k in ('image_files','mapping','true_rule')):
                raise ValueError('Paired evaluation inputs differ')
        report[condition]['delta_from_pretrained']=report[condition]['accuracy']-report['pretrained']['accuracy']
    report['between_budgets']={f'{b}_minus_{a}':report[str(b)]['accuracy']-report[str(a)]['accuracy'] for a,b in ((24,45),(45,90),(24,90))}
    report['limitations']=plan['limitations'];report['previous_run']=plan['previous_100_percent']
    out=Path(root)/f'seed-{seed}'/'comparison'
    out.mkdir(parents=True,exist_ok=True)
    write(out/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for n in BUDGETS:
        rows=[json.loads(l) for l in (output_path(root,seed,str(n))/'losses.jsonl').read_text().splitlines()]
        plt.plot([r['step'] for r in rows],[r['training_loss'] for r in rows],alpha=.4,label=f'{n} train')
        val=[r for r in rows if 'validation_loss' in r]
        plt.plot([r['step'] for r in val],[r['validation_loss'] for r in val],'o-',label=f'{n} validation')
    plt.xlabel('Successful optimizer updates');plt.ylabel('Answer-token loss');plt.legend();plt.tight_layout()
    plt.savefig(out/'loss_curves.png');plt.close()
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition',choices=('pretrained',*map(str,BUDGETS)),default='pretrained')
    p.add_argument('--steps',type=int,default=46)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--subset-seed',type=int,default=2026)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--compare',action='store_true')
    args=p.parse_args()
    original=source(args.source_repo)
    manifest=original.load(args.manifest,args.data_dir)
    plan=make_plan(manifest,args.steps,args.seed,args.subset_seed)
    for protected in (args.source_repo,args.data_dir):
        if args.out.resolve()==protected.resolve() or protected.resolve() in args.out.resolve().parents:
            p.error('Output root must be separate from source checkout and dataset')
    if args.out.resolve()==args.manifest.parent.resolve():
        p.error('Do not use the original manifest directory as the output root')
    if args.validate_only:
        print(json.dumps({**plan,'conditions':{k:{a:b for a,b in v.items() if a not in ('update_schedule','image_reuse_counts')} for k,v in plan['conditions'].items()}},indent=2))
        return
    if args.compare:
        compare_outputs(args.out,args.seed,plan);return
    if args.steps<2 and args.smoke_test:
        p.error('Smoke test needs --steps at least 2')
    trainer,common=source(args.source_repo,gpu=True)
    condition=plan['conditions'].get(args.condition)
    config=dict(plan_hash=digest(plan),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_hash=MANIFEST_HASH,source_commit=SOURCE_COMMIT,condition=condition,
        training_config=condition['training_config'] if condition else None,
        model_settings=common.settings(SimpleNamespace(revision=REVISION,precision='4bit',max_pixels=100352)),
        software=common.versions(),seed=args.seed,smoke_test=args.smoke_test,partition='validation')
    path=output_path(args.out,args.seed,args.condition,args.smoke_test)
    with locked_output(path,config,args.resume):
        write(path/'plan.json',plan)
        if args.condition!='pretrained':
            baseline=output_path(args.out,args.seed,'pretrained')
            if not args.smoke_test:
                base_config=read(baseline/'config.json')
                if base_config['plan_hash']!=digest(plan) or read(baseline/'metrics.json')['trials']!=90 or any(base_config[k]!=config[k] for k in ('software','model_settings','runner_sha256')):
                    raise ValueError('Run the matched pretrained validation condition first with identical software/model/runner')
            train(args,manifest,plan,config,path,trainer,common)
        params=SimpleNamespace(revision=REVISION,precision='4bit',max_pixels=100352,
            adapter=path/'best_adapter' if args.condition!='pretrained' else None)
        # Release training/model memory before reloading selected weights for readout.
        import gc
        import torch
        gc.collect();torch.cuda.empty_cache()
        adapter=common.load_model(params)
        items=[s for s in manifest['sets'] if s['partition']=='validation']
        if args.smoke_test:
            items=[next(s for s in items if s['rule']==r) for r in RULES]
        write(path/'metrics.json',metrics(evaluate(adapter,common,items,args.data_dir,path)))
        print('Output:',path)


if __name__=='__main__':
    main()
