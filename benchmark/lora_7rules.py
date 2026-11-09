"""Seven generating rules, ten images, frozen pretrained versus QLoRA Qwen.

All artifacts go under a new --out directory (normally Google Drive). No imports
from historical branches; shared inference comes from compare_3_vs_7. Default
readout is validation only. final-test explicitly freezes selection first.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import build_sets as builder
import compare_3_vs_7 as comparison

SEED = 2026
COUNTS = {'train': 21, 'validation': 7, 'test': 14}
LABELS = list('1234567')
MAX_GRADIENT_RETRIES = 4  # Initial attempt plus at most four scale-reduced replays.
HISTORICAL_HASH = '701936daac921fd19da203fccda2524d43c7277999ebdee0412041d987f1fcd4'
MODEL = dict(model=comparison.shared.MODEL, revision=comparison.shared.REVISION,
    quantization='4-bit NF4', double_quantization=True, compute_dtype='float16',
    attention='sdpa', thinking=False, max_pixels=128*28*28, images_per_input=10,
    scoring='next-token likelihood restricted/normalized over numeric 1-7',
    training_objective='full-vocabulary cross entropy on the single answer token only')
TRAIN = dict(rank=8, alpha=16, dropout=.05, learning_rate=5e-5, epochs=3,
    optimizer='AdamW', weight_decay=.01, betas=[.9,.999], eps=1e-8,
    microbatch=1, gradient_accumulation=4, seed=SEED, initial_gradient_scale=8.,
    gradient_clip=1., max_tokens=16000, validation='end of epoch; fixed balanced rotation per set',
    checkpoint='latest at every successful optimizer boundary',
    selection='lowest validation full-vocabulary answer-token loss',
    initialization='fresh pretrained base and default PEFT LoRA initialization, never the old adapter')
CAUTION = ('Three operands and seven generating rules. Not an exact replication of Roshan: '
    'his generation code and exact learning rate are unavailable; differences do not isolate operand count. '
    'Test images were previously exposed through exploratory experiments. Sets reuse images within '
    'partitions and rotations repeat the same sets, so neither trials nor sets are fully independent. '
    'Positive-only diagnostic shortcuts are possible; better classification does not prove Boolean reasoning '
    'or generalization to unseen logical configurations.')
RULES_PATH = Path(__file__).with_name('rules_7.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def dataset_identity(data):
    data=Path(data)
    records=read(data/'metadata.json')['records']
    if len(records)!=1000 or len({r['filename'] for r in records})!=1000:
        raise ValueError('Expected exactly 1,000 unique source images')
    if {p.name for p in (data/'images').glob('*.png')}!={r['filename'] for r in records}:
        raise ValueError('Image filenames differ from source metadata')
    for row in records:
        name=row['filename'];bits=row['bits']
        if Path(name).name!=name or len(bits)!=3 or any(type(x) is not int or x not in (0,1) for x in bits):
            raise ValueError(f'Invalid source filename/configuration: {name}')
        if row['split'] not in ('train','test'):
            raise ValueError('Expected historical train/test metadata')
    if Counter(tuple(r['bits']) for r in records)!={bits:125 for bits in product((0,1),repeat=3)}:
        raise ValueError('Expected 125 source images per A/B/C configuration')
    h=hashlib.sha256((data/'metadata.json').read_bytes())
    for path in sorted((data/'images').glob('*.png')):
        h.update(path.name.encode());h.update(path.read_bytes())
    return records,h.hexdigest()


def validate_partitions(records, assignments):
    if set(assignments)!={r['filename'] for r in records}:
        raise ValueError('Every source image must have exactly one partition assignment')
    if Counter(assignments.values())!={'train':800,'validation':100,'test':100}:
        raise ValueError('Required source-image split is 800/100/100')
    config_counts={}
    for index,bits in enumerate(product((0,1),repeat=3)):
        rows=[r for r in records if tuple(r['bits'])==bits]
        original=Counter(r['split'] for r in rows)
        expected={'train':100,'validation':12+index%2,'test':13-index%2}
        actual=Counter(assignments[r['filename']] for r in rows)
        if original!={'train':100,'test':25} or actual!=expected:
            raise ValueError(f'Incorrect configuration-stratified partition for {bits}: {dict(actual)}')
        if any((r['split']=='train')!=(assignments[r['filename']]=='train') for r in rows):
            raise ValueError('Source training images may not cross into held-out partitions')
        config_counts[''.join(map(str,bits))]=dict(actual)
    return config_counts


def partitions(data, historical=None):
    records,identity=dataset_identity(data)
    if historical is None:
        candidate=Path(data).parent/'results/04-simple-lora/rule_identification_manifest.json'
        if candidate.is_file():historical=candidate
    if historical is not None and Path(historical).is_file():
        history=read(historical)
        if digest(history)!=HISTORICAL_HASH or history.get('dataset_fingerprint')!=identity:
            raise ValueError('Historical manifest hash or complete source-dataset fingerprint differs')
        assignments=history['image_partitions']
        provenance=dict(method='verified exact historical image_partitions',
            historical_manifest_sha256=file_hash(historical),historical_manifest_canonical_hash=digest(history))
    else:
        assignments={}
        rng=random.Random(SEED)
        for index,bits in enumerate(product((0,1),repeat=3)):
            train=sorted(r['filename'] for r in records if tuple(r['bits'])==bits and r['split']=='train')
            held=sorted(r['filename'] for r in records if tuple(r['bits'])==bits and r['split']=='test')
            rng.shuffle(held)
            assignments.update({name:'train' for name in train})
            assignments.update({name:'validation' if i<12+index%2 else 'test' for i,name in enumerate(held)})
        provenance=dict(method='reconstructed historical seed-2026 procedure; not verified identical to a supplied historical manifest')
    counts=validate_partitions(records,assignments)
    return assignments,dict(provenance,source_dataset_fingerprint=identity,
        source_metadata_sha256=file_hash(Path(data)/'metadata.json'),configuration_image_counts=counts)


def generate(data, historical=None):
    spec=comparison.validate_rules(RULES_PATH)
    assignments,provenance=partitions(data,historical)
    # Existing tested builder reads partition bytes. The temporary file is never
    # a replacement for historical input; only the final new mapping is persisted.
    mapping_bytes=(json.dumps(assignments,indent=2)+'\n').encode()
    with tempfile.TemporaryDirectory() as folder:
        mapping_path=Path(folder)/'partitions.json';mapping_path.write_bytes(mapping_bytes)
        manifest=builder.make_manifest(data,RULES_PATH,mapping_path,COUNTS,10,SEED)
    for split,count in COUNTS.items():
        for rule in comparison.IDS:
            items=[s for s in manifest['sets'] if s['split']==split and s['rule']==rule]
            if len(items)!=count:
                raise ValueError('Unexpected set count')
            for index,item in enumerate(items):
                # Cyclic mappings preserve prompt meaning while balancing labels.
                label=index%7
                ri=comparison.IDS.index(rule)
                item['fixed_rotation']=(ri-label)%7
                item['option_mapping']=comparison.mapping(7,item['fixed_rotation'])
                item['target_label']=str(label+1)
    manifest.update(protocol='seven-generating-rules-qlora-v1',status='fixed experimental protocol; no GPU outcomes claimed',rules=spec,provenance=provenance,
        image_partitions=assignments,partitions_canonical_hash=digest(assignments),caution=CAUTION)
    report=validate_manifest(manifest,data)
    manifest['validation_report']=report
    return manifest,mapping_bytes


def validate_manifest(manifest,data):
    spec=comparison.validate_rules(RULES_PATH)
    records,identity=dataset_identity(data)
    if manifest['provenance']['source_dataset_fingerprint']!=identity or manifest['source_metadata_sha256']!=file_hash(Path(data)/'metadata.json'):
        raise ValueError('Saved manifest uses a different source dataset')
    if manifest['rules']!=spec or manifest['rules_sha256']!=file_hash(RULES_PATH):
        raise ValueError('Saved rule definitions/hash differ')
    assignments=manifest['image_partitions']
    validate_partitions(records,assignments)
    expected_mapping=(json.dumps(assignments,indent=2)+'\n').encode()
    if manifest['partitions_sha256']!=hashlib.sha256(expected_mapping).hexdigest() or manifest['partitions_canonical_hash']!=digest(assignments):
        raise ValueError('Partition hash differs')
    if manifest['seed']!=SEED or manifest['images_per_set']!=10 or manifest['sets_per_rule_by_partition']!=COUNTS:
        raise ValueError('Manifest settings differ from the fixed protocol')
    if len(manifest['sets'])!=294 or manifest['set_count']!=294:
        raise ValueError('Expected exactly 294 image sets')
    by_name={r['filename']:r for r in records}
    seen_ids=set();identities=set();usage=defaultdict(Counter);counts=Counter();positions=defaultdict(Counter)
    configuration_counts=defaultdict(Counter)
    for item in manifest['sets']:
        sid,split,rule=item['set_id'],item['split'],item['rule']
        if sid in seen_ids or split not in COUNTS or rule not in comparison.IDS:
            raise ValueError(f'Duplicate or invalid set ID/split/rule: {sid}')
        seen_ids.add(sid)
        names,bits=item['image_files'],item['bits']
        if len(names)!=10 or len(set(names))!=10 or len(bits)!=10:
            raise ValueError(f'{sid}: ten distinct ordered images are required')
        identity_key=tuple(sorted(names))
        if identity_key in identities:
            raise ValueError(f'{sid}: duplicate unordered image-set identity')
        identities.add(identity_key)
        for name,bit in zip(names,bits):
            if name not in by_name or assignments[name]!=split or bit!=''.join(map(str,by_name[name]['bits'])):
                raise ValueError(f'{sid}: missing, mislabelled or cross-partition image {name}')
        table=next(r['true_bits'] for r in spec['rules'] if r['id']==rule)
        if set(bits)!=set(table):
            raise ValueError(f'{sid}: missing coverage or nonsatisfying configuration')
        matches=builder.compatible_rules(bits,spec['rules'])
        if matches!=[rule]:
            raise ValueError(f'{sid}: not uniquely identifiable; compatible={matches}')
        if item['option_mapping']!=comparison.mapping(7,item['fixed_rotation']) or item['option_mapping'][item['target_label']]!=rule:
            raise ValueError(f'{sid}: incorrect answer mapping')
        counts[(split,rule)]+=1;positions[(split,rule)][item['target_label']]+=1
        usage[split].update(names);configuration_counts[(split,rule)].update(bits)
    report={}
    for split,n in COUNTS.items():
        for rule in comparison.IDS:
            if counts[(split,rule)]!=n or positions[(split,rule)]!={str(i):n//7 for i in range(1,8)}:
                raise ValueError(f'{split}/{rule}: set counts or answer positions are not balanced')
        report[split]=dict(sets=n*7,sets_per_rule=n,source_partition_images=sum(v==split for v in assignments.values()),
            unique_used_images=len(usage[split]),image_appearances=sum(usage[split].values()),
            reused_source_images=sum(v>1 for v in usage[split].values()),image_reuse_counts=dict(usage[split]),
            configurations_per_rule={r:dict(configuration_counts[(split,r)]) for r in comparison.IDS},
            correct_answer_positions_per_rule={r:dict(positions[(split,r)]) for r in comparison.IDS})
    if any(set(usage[a])&set(usage[b]) for a in usage for b in usage if a!=b):
        raise ValueError('Cross-partition image leakage')
    return report


def load_manifest(root,data):
    path=Path(root)/'dataset/manifest.json'
    manifest=read(path)
    if validate_manifest(manifest,data)!=manifest['validation_report']:
        raise ValueError('Saved validation report differs')
    if (Path(root)/'dataset/partitions.json').read_bytes()!=(json.dumps(manifest['image_partitions'],indent=2)+'\n').encode():
        raise ValueError('Saved partition file differs from manifest')
    return manifest


def training_schedule(manifest,smoke=False):
    items=[s for s in manifest['sets'] if s['split']=='train']
    result=[]
    for epoch in range(3):
        order=items.copy();random.Random(SEED+epoch).shuffle(order)
        for start in range(0,len(order),4):
            result.append(dict(epoch=epoch+1,set_ids=[s['set_id'] for s in order[start:start+4]],
                epoch_end=start+4>=len(order)))
    return result[:1] if smoke else result


def code_identity():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
            ('lora_7rules.py','compare_3_vs_7.py','pilot_10_vs_20.py','build_sets.py','rules_7.json')}


def configuration(manifest,kind,partition='validation',adapter=None,smoke=False):
    software={}
    for name in ('torch','transformers','peft','bitsandbytes','accelerate','pillow'):
        try:software[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:software[name]=None
    return dict(protocol=manifest['protocol'],manifest_hash=digest(manifest),model=MODEL,
        training=TRAIN,kind=kind,partition=partition,adapter=adapter,smoke_test=smoke,
        rules=manifest['rules'],prompts=[comparison.prompt_for(manifest['rules'],comparison.mapping(7,r)) for r in range(7)],
        source_hashes=code_identity(),software=software,caution=CAUTION)


@contextmanager
def output(path,config,resume=False):
    import fcntl
    path=Path(path)
    if path.exists() and not resume:
        raise ValueError(f'Output exists: {path}; explicitly --resume or use a new experiment root')
    if resume and not path.exists():
        raise ValueError('Cannot resume nonexistent output')
    path.mkdir(parents=True,exist_ok=True)
    with (path/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Another process is using this output') from exc
        if resume:
            if read(path/'config.json')!=config:raise ValueError('Incompatible existing output configuration')
        else:atomic_json(path/'config.json',config)
        yield path


def prepare(model,processor,item,data,prompt,token_ids):
    """Exactly the comparison runner's ordered RGB/resize/chat preprocessing."""
    from PIL import Image
    if len(item['image_files'])!=10:
        raise ValueError('Exactly ten images are required; no input truncation is allowed')
    content=[]
    for name in item['image_files']:
        with Image.open(Path(data)/'images'/name) as file:picture=file.convert('RGB')
        if picture.width*picture.height>MODEL['max_pixels']:
            scale=(MODEL['max_pixels']/(picture.width*picture.height))**.5
            picture=picture.resize((max(1,int(picture.width*scale)),max(1,int(picture.height*scale))),Image.Resampling.LANCZOS)
        content.append(dict(type='image',image=picture))
    content.append(dict(type='text',text=prompt));messages=[dict(role='user',content=content)]
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=processor.tokenizer.encode(rendered,add_special_tokens=False)
    for label,tid in zip(LABELS,token_ids):
        if processor.tokenizer.encode(rendered+label,add_special_tokens=False)!=prefix+[tid]:
            raise ValueError(f'Answer label {label} is not an append-only single token at the actual assistant boundary')
    prepared=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_dict=True,return_tensors='pt').to(model.device)
    if prepared['input_ids'].shape[1]+1>TRAIN['max_tokens']:
        raise ValueError('Input exceeds the token guard; no image dropped, resolution lowered or text truncated')
    return prepared


def answer_loss(model,prepared,target):
    import torch
    logits=model(**prepared,use_cache=False,logits_to_keep=1).logits[0,-1].float()
    return torch.nn.functional.cross_entropy(logits[None],torch.tensor([target],device=logits.device))


def decoder_targets(model):
    import torch
    layers=None
    for path in ('model.language_model','model.language_model.model','model.text_model','model'):
        obj=model
        for name in path.split('.'):obj=getattr(obj,name,None)
        if hasattr(obj,'layers'):layers=obj.layers;break
    if layers is None:raise ValueError('Cannot locate decoder layers')
    names={id(module):name for name,module in model.named_modules()}
    selected=[]
    for block in layers:
        found=[names[id(block)]+'.'+name for name,module in block.named_modules()
               if name and isinstance(module,torch.nn.Linear)]
        if not found:raise ValueError('Decoder layer has no supported linear targets')
        selected.extend(found)
    if len(set(selected))!=len(selected):raise ValueError('Duplicate decoder targets')
    return selected


def trainable_inventory(model,targets):
    parameters={name:p.numel() for name,p in model.named_parameters() if p.requires_grad}
    if not parameters or any('lora_' not in name or not any(t+'.' in name for t in targets) for name in parameters):
        raise ValueError('Trainable parameters escape selected decoder LoRA modules')
    return dict(target_modules=targets,target_count=len(targets),parameters=parameters,total=sum(parameters.values()))


def optimizer_update(model,optimizer,scaler):
    import torch
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    before={n:p.detach().cpu().clone() for n,p in parameters if 'lora_B' in n}
    if not before:raise ValueError('No trainable LoRA B matrices')
    scale=scaler.get_scale();scaler.unscale_(optimizer)
    active=[p for _,p in parameters if p.grad is not None]
    if not active:raise ValueError('No trainable gradients')
    bad={name:int((~torch.isfinite(p.grad)).sum().item()) for name,p in parameters if p.grad is not None}
    bad={name:count for name,count in bad.items() if count}
    finite=not bad
    norm=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)) if finite else None
    called=[];hook=optimizer.register_step_post_hook(lambda *_:called.append(True))
    try:
        if finite or scaler.is_enabled():scaler.step(optimizer)
        scaler.update()
    finally:hook.remove()
    changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in parameters if n in before)
    if not all(bool(torch.isfinite(p).all()) for _,p in parameters):raise ValueError('Non-finite adapter weights')
    return dict(optimizer_step_occurred=bool(called),finite_gradients=finite,gradient_norm_before_clipping=norm,
        lora_b_matrices=len(before),lora_b_matrices_changed=changed,gradient_scale_before=scale,
        gradient_scale_after=scaler.get_scale(),nonfinite_gradient_values=sum(bad.values()),
        nonfinite_gradient_parameter_names=list(bad),nonfinite_gradient_counts_by_parameter=bad)


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def batch_update(model,optimizer,scaler,batch,loss_fn,path,planned_update,
                 max_retries=MAX_GRADIENT_RETRIES):
    """Replay only an untouched, AMP-skipped nonfinite batch at a reduced scale.

    Every replay starts from the same Python/NumPy/CPU/CUDA RNG snapshot. Keep
    GradScaler's state after its backoff, not the original scale. Only a successful
    return may advance the training cursor or produce an optimizer checkpoint.
    """
    import torch
    if not batch or max_retries<0:raise ValueError('Need a nonempty batch and nonnegative retry limit')
    path=Path(path);initial_rng=capture_rng();attempts=[];succeeded=False
    def failure(reason):
        diagnostic=dict(planned_update=planned_update,set_ids=[s['set_id'] for s in batch],
            reason=reason,max_retries=max_retries,retry_count=max(0,len(attempts)-1),
            attempted_scales=[a['gradient_scale_before'] for a in attempts],attempts=attempts)
        atomic_json(path/'update_failure.json',diagnostic)
        raise RuntimeError('No successful-update credit: '+json.dumps(diagnostic,allow_nan=False))
    try:
        for retry in range(max_retries+1):
            # In particular, discard all invalid gradients before another backward.
            optimizer.zero_grad(set_to_none=True)
            restore_rng(initial_rng)
            model.train();value=0.
            for item in batch:
                with torch.autocast('cuda',dtype=torch.float16,enabled=torch.cuda.is_available()):
                    one=loss_fn(item)
                if not torch.isfinite(one):failure('Non-finite forward loss; not an AMP-gradient retry')
                value+=float(one.detach())/len(batch)
                scaler.scale(one/len(batch)).backward()
            evidence=optimizer_update(model,optimizer,scaler)
            attempt=dict(attempt=retry,**evidence);attempts.append(attempt)
            append(path/'update_attempts.jsonl',dict(planned_update=planned_update,
                set_ids=[s['set_id'] for s in batch],**attempt))
            valid=evidence['optimizer_step_occurred'] and evidence['finite_gradients'] and evidence['lora_b_matrices_changed']>0
            if valid:
                succeeded=True
                return value,dict(evidence,retry_count=retry,
                    attempted_scales=[a['gradient_scale_before'] for a in attempts],attempts=attempts,
                    attempted_presentations=len(batch)*len(attempts))
            # Retrying after any actual optimizer mutation would be unsafe.
            eligible=(not evidence['finite_gradients'] and not evidence['optimizer_step_occurred']
                and evidence['lora_b_matrices_changed']==0 and scaler.is_enabled()
                and math.isfinite(evidence['gradient_scale_after'])
                and 0<evidence['gradient_scale_after']<evidence['gradient_scale_before'])
            if not eligible:failure('Unsafe/non-recoverable update: retry requires unchanged weights, a skipped step and reduced AMP scale')
            if retry==max_retries:failure('Non-finite gradients persist after bounded AMP retries')
            print(f"update {planned_update}: retry {retry+1}/{max_retries}, AMP scale "
                  f"{evidence['gradient_scale_before']} -> {evidence['gradient_scale_after']}; "
                  f"nonfinite gradients={evidence['nonfinite_gradient_values']}",flush=True)
    finally:
        if not succeeded:
            optimizer.zero_grad(set_to_none=True)
            restore_rng(initial_rng)


def metrics(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['set_id']].append(row)
    if not groups or any(sorted(r['rotation'] for r in group)!=list(range(7)) for group in groups.values()):
        raise ValueError('Metrics require seven rotations for every included set')
    n=len(rows);accuracy=sum(r['correct'] for r in rows)/n
    return dict(sets=len(groups),evaluations=n,accuracy=accuracy,chance=1/7,
        improvement_over_chance=accuracy-1/7,chance_adjusted_accuracy=(accuracy-1/7)/(1-1/7),
        per_rule={rule:sum(r['correct'] for r in rows if r['true_rule']==rule)/sum(r['true_rule']==rule for r in rows) for rule in comparison.IDS},
        confusion={rule:{pred:sum(r['true_rule']==rule and r['predicted_rule']==pred for r in rows) for pred in comparison.IDS} for rule in comparison.IDS},
        answer_position_counts={label:sum(r['predicted_label']==label for r in rows) for label in LABELS},
        rotation_consistency=sum(len({r['predicted_rule'] for r in g})==1 for g in groups.values())/len(groups),
        mean_correct_rule_probability=sum(r['probabilities'][r['true_label']] for r in rows)/n,
        mean_restricted_answer_nll=sum(-math.log(max(r['probabilities'][r['true_label']],1e-30)) for r in rows)/n,
        caution=CAUTION)


def evaluation_jobs(manifest,partition,smoke=False):
    items=[s for s in manifest['sets'] if s['split']==partition]
    if smoke:items=[next(s for s in items if s['rule']==rule) for rule in comparison.IDS]
    result={}
    for item in items:
        for rotation in range(7):
            mapping=comparison.mapping(7,rotation);key=f"{item['set_id']}:{rotation}"
            result[key]=dict(key=key,set_id=item['set_id'],true_rule=item['rule'],rotation=rotation,
                image_files=item['image_files'],bits=item['bits'],option_mapping=mapping,
                prompt=comparison.prompt_for(manifest['rules'],mapping),
                true_label=next(l for l,r in mapping.items() if r==item['rule']))
    return result


def load_rows(path,jobs):
    path=Path(path)
    if not path.exists():return {}
    content=path.read_text()
    if content and not content.endswith('\n'):raise ValueError('Partial prediction log line; preserve and inspect before resuming')
    rows={}
    for line in content.splitlines():
        row=json.loads(line);key=row['key']
        if key in rows or key not in jobs:raise ValueError('Duplicate or unexpected prediction key')
        comparison.validate_row(row,jobs[key]);rows[key]=row
    return rows


def append(path,row):
    with Path(path).open('a') as file:
        file.write(json.dumps(row,allow_nan=False)+'\n');file.flush();os.fsync(file.fileno())


def adapter_identity(root):
    root=Path(root)
    return {name:file_hash(root/name) for name in ('adapter_config.json','adapter_model.safetensors','experiment.json')}


def record_tokens(path,token_ids):
    if len(token_ids)!=7 or len(set(token_ids))!=7 or any(type(t) is not int or t<0 for t in token_ids):
        raise ValueError('Invalid seven-label tokenizer representations')
    path=Path(path)/'answer_tokens.json'
    value=dict(zip(LABELS,token_ids))
    if path.exists() and read(path)!=value:raise ValueError('Actual answer token IDs differ from saved readout')
    if not path.exists():atomic_json(path,value)


def evaluate(root,data,manifest,kind='pretrained',partition='validation',resume=False,smoke=False):
    root=Path(root)
    adapter_path=root/'training/best_adapter' if kind=='finetuned' else None
    if adapter_path:
        require_selection(root,manifest)
    identity=adapter_identity(adapter_path) if adapter_path else None
    config=configuration(manifest,kind,partition,identity,smoke)
    path=root/('final-test' if partition=='test' else 'evaluations')/kind/partition
    if smoke:path=root/'smoke-evaluation'/kind
    jobs=evaluation_jobs(manifest,partition,smoke)
    with output(path,config,resume):
        rows=load_rows(path/'predictions.jsonl',jobs)
        if len(rows)<len(jobs):
            model,processor,_=comparison.shared.load_model()
            if adapter_path:
                from peft import PeftModel
                model=PeftModel.from_pretrained(model,adapter_path,is_trainable=False).eval()
            tokens=comparison.label_tokens(processor.tokenizer)
            record_tokens(path,tokens)
            for key,job in jobs.items():
                if key in rows:continue
                import torch
                prepared=prepare(model,processor,job,data,job['prompt'],tokens)
                with torch.inference_mode():
                    logits=model(**prepared,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                    scores=logits[tokens]
                    if not torch.isfinite(scores).all():raise ValueError('Non-finite answer logits')
                    probs=dict(zip(LABELS,scores.softmax(0).cpu().tolist()))
                n=int(prepared['input_ids'].shape[1])
                label=max(LABELS,key=probs.get);pred=job['option_mapping'][label]
                row=dict(job,probabilities=probs,predicted_label=label,predicted_rule=pred,
                    correct=pred==job['true_rule'],input_tokens=n)
                comparison.validate_row(row,job);append(path/'predictions.jsonl',row);rows[key]=row
                print(f'{len(rows)}/{len(jobs)} {key}: {label}',flush=True)
            del model,processor
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        report=metrics(list(rows.values()));atomic_json(path/'metrics.json',report)
    return report


def save_checkpoint(path,model,processor,state):
    import torch
    path=Path(path);temp=path/'latest.tmp';latest=path/'latest';previous=path/'latest.previous'
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir();model.save_pretrained(temp);processor.save_pretrained(temp)
    torch.save(state,temp/'training_state.pt')
    if previous.exists():shutil.rmtree(previous)
    if latest.exists():latest.rename(previous)
    temp.rename(latest)
    if previous.exists():shutil.rmtree(previous)


def checkpoint_state(path,config_hash):
    import torch
    path=Path(path);latest=path/'latest'
    if not latest.exists() and (path/'latest.previous').exists():(path/'latest.previous').rename(latest)
    if not latest.exists():return None
    # Only trusted optimizer checkpoints produced by this implementation.
    state=torch.load(latest/'training_state.pt',map_location='cpu',weights_only=False)
    if state['config_hash']!=config_hash:raise ValueError('Checkpoint/configuration mismatch')
    if state['next_update']!=len(state['losses']) or any(r['update']!=i+1 or not r['optimizer_step_occurred'] for i,r in enumerate(state['losses'])):
        raise ValueError('Inconsistent successful-update checkpoint journal')
    return state


def require_selection(root,manifest):
    root=Path(root);path=root/'training';config=read(path/'config.json')
    expected=configuration(manifest,'training')
    if config!=expected:raise ValueError('Training model/manifest/preprocessing/software differs from evaluation')
    complete=read(path/'complete.json');selection=read(path/'selection.json')
    if complete['successful_updates']!=111 or complete['presentations']!=441 or complete['smoke_test'] or complete['selected_epoch']!=selection['epoch']:
        raise ValueError('Need a completed 111-update, 441-presentation selected model')
    if read(path/'best_adapter/experiment.json')['config_hash']!=digest(config):raise ValueError('Selected adapter has wrong provenance')
    losses=read(path/'losses.json')
    schedule=training_schedule(manifest)
    if len(losses)!=111 or any(r['update']!=i+1 or r['epoch']!=schedule[i]['epoch'] or r['set_ids']!=schedule[i]['set_ids'] or r['examples']!=len(schedule[i]['set_ids']) or not math.isfinite(r['training_loss']) or not r['optimizer_step_occurred'] or not r['finite_gradients'] or r['lora_b_matrices_changed']<1 for i,r in enumerate(losses)):
        raise ValueError('Missing successful-training update evidence')
    validations=read(path/'validation_losses.json')
    best=min(validations,key=lambda r:r['loss'])
    if [v['epoch'] for v in validations]!=[1,2,3] or any(v['sets']!=49 or not math.isfinite(v['loss']) for v in validations) or selection['epoch']!=best['epoch'] or selection['loss']!=best['loss']:
        raise ValueError('Selected adapter is not the minimum validation-loss epoch')
    if complete['adapter_identity']!=adapter_identity(path/'best_adapter'):
        raise ValueError('Selected adapter files differ from training completion')
    return selection


def train(root,data,manifest,resume=False,smoke=False):
    root=Path(root)
    if (root/'selection_freeze.json').exists():raise ValueError('Final-test selection is frozen; training cannot alter this experiment')
    config=configuration(manifest,'training',smoke=smoke)
    if not smoke:
        base=root/'evaluations/pretrained/validation'
        if read(base/'config.json')!=configuration(manifest,'pretrained') or read(base/'metrics.json')['evaluations']!=343:
            raise ValueError('Run the matching full seven-rule pretrained validation baseline first')
        if len(load_rows(base/'predictions.jsonl',evaluation_jobs(manifest,'validation')))!=343:
            raise ValueError('Incomplete baseline predictions')
    path=root/('smoke-training' if smoke else 'training')
    with output(path,config,resume):
        import torch
        from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training,get_peft_model_state_dict,set_peft_model_state_dict
        import numpy as np
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        model,processor,_=comparison.shared.load_model()
        token_ids=comparison.label_tokens(processor.tokenizer)
        record_tokens(path,token_ids)
        targets=decoder_targets(model)
        for p in model.parameters():p.requires_grad=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False
        state=checkpoint_state(path,digest(config)) if resume else None
        if state:model=PeftModel.from_pretrained(model,path/'latest',is_trainable=True)
        else:model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=targets))
        inventory=trainable_inventory(model,targets)
        if state and read(path/'trainable.json')!=inventory:raise ValueError('Resolved LoRA targets/trainable parameters changed')
        atomic_json(path/'trainable.json',inventory)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,
            betas=(.9,.999),eps=1e-8,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=8.)
        losses=[];validations=[];best_weights=None;selection=None;start=0
        if state:
            optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            if 'rng_state' in state:restore_rng(state['rng_state'])
            else:
                torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            losses=state['losses'];validations=state['validations'];best_weights=state['best_weights'];selection=state['selection'];start=state['next_update']
        jobs=training_schedule(manifest,smoke);by_id={s['set_id']:s for s in manifest['sets']}
        if state and (start>len(jobs) or any(r['epoch']!=jobs[i]['epoch'] or r['set_ids']!=jobs[i]['set_ids'] or r['examples']!=len(jobs[i]['set_ids']) for i,r in enumerate(losses))):
            raise ValueError('Saved optimizer cursor/data order differs from regenerated schedule')
        val=[s for s in manifest['sets'] if s['split']=='validation']
        if smoke:val=[next(s for s in val if s['rule']==r) for r in comparison.IDS]
        def loss(item):
            prompt=comparison.prompt_for(manifest['rules'],item['option_mapping'])
            prepared=prepare(model,processor,item,data,prompt,token_ids)
            return answer_loss(model,prepared,token_ids[LABELS.index(item['target_label'])])
        for index in range(start,len(jobs)):
            job=jobs[index]
            batch=[by_id[sid] for sid in job['set_ids']]
            value,evidence=batch_update(model,optimizer,scaler,batch,loss,path,index+1)
            record=dict(update=index+1,epoch=job['epoch'],set_ids=job['set_ids'],examples=len(batch),training_loss=value,**evidence)
            losses.append(record)
            if job['epoch_end'] or smoke:
                model.eval()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    v=sum(float(loss(s)) for s in val)/len(val)
                if not math.isfinite(v):raise ValueError('Non-finite validation loss')
                validations.append(dict(epoch=job['epoch'],loss=v,sets=len(val),mode='fixed position-balanced options' if not smoke else 'smoke subset, not checkpoint evidence'))
                if selection is None or v<selection['loss']:
                    selection=dict(epoch=job['epoch'],loss=v,criterion=TRAIN['selection'])
                    best_weights={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            state=dict(config_hash=digest(config),next_update=index+1,losses=losses,validations=validations,
                best_weights=best_weights,selection=selection,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),rng_state=capture_rng())
            save_checkpoint(path,model,processor,state)
            atomic_json(path/'losses.json',losses);atomic_json(path/'validation_losses.json',validations)
            print(f"update {index+1}/{len(jobs)} epoch={job['epoch']} examples={len(batch)} changed_B={evidence['lora_b_matrices_changed']}",flush=True)
        # Checkpoint is authoritative if interruption occurred during mirror writes.
        atomic_json(path/'losses.json',losses);atomic_json(path/'validation_losses.json',validations)
        set_peft_model_state_dict(model,best_weights)
        model.save_pretrained(path/'best_adapter');processor.save_pretrained(path/'best_adapter')
        atomic_json(path/'best_adapter/experiment.json',dict(config_hash=digest(config),selection=selection))
        atomic_json(path/'selection.json',selection)
        atomic_json(path/'complete.json',dict(successful_updates=len(losses),presentations=sum(r['examples'] for r in losses),
            attempted_presentations=sum(r['attempted_presentations'] for r in losses),
            selected_epoch=selection['epoch'],smoke_test=smoke,test_used=False,adapter_identity=adapter_identity(path/'best_adapter')))
        print('Saved:',path)


def compare_results(root,manifest,partition):
    root=Path(root);prefix='final-test' if partition=='test' else 'evaluations'
    reports={};rows_by_kind={}
    for kind in ('pretrained','finetuned'):
        path=root/prefix/kind/partition
        expected=configuration(manifest,kind,partition,adapter_identity(root/'training/best_adapter') if kind=='finetuned' else None)
        if read(path/'config.json')!=expected:raise ValueError('Comparison configuration mismatch')
        jobs=evaluation_jobs(manifest,partition);rows=load_rows(path/'predictions.jsonl',jobs)
        if set(rows)!=set(jobs):raise ValueError('Incomplete paired evaluation')
        rows_by_kind[kind]=rows;reports[kind]=metrics(list(rows.values()))
    baseline_tokens=read(root/prefix/'pretrained'/partition/'answer_tokens.json')
    if baseline_tokens!=read(root/prefix/'finetuned'/partition/'answer_tokens.json') or baseline_tokens!=read(root/'training/answer_tokens.json'):
        raise ValueError('Training and before/after answer token IDs differ')
    before,after=rows_by_kind['pretrained'],rows_by_kind['finetuned']
    for key,row in before.items():
        if any(row[k]!=after[key][k] for k in ('image_files','bits','option_mapping','true_rule','prompt')):
            raise ValueError('Paired inputs differ')
    reports['accuracy_difference']=reports['finetuned']['accuracy']-reports['pretrained']['accuracy']
    reports['changed_predictions']=sum(before[k]['predicted_rule']!=after[k]['predicted_rule'] for k in before)
    reports['caution']=CAUTION
    path=root/'comparison'/partition;path.mkdir(parents=True,exist_ok=True)
    atomic_json(path/'summary.json',reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    losses=read(root/'training/losses.json');vals=read(root/'training/validation_losses.json')
    fig,ax=plt.subplots();ax.plot([r['update'] for r in losses],[r['training_loss'] for r in losses],label='train')
    ax.plot([r['epoch']*37 for r in vals],[r['loss'] for r in vals],'o-',label='validation')
    ax.set(xlabel='Successful optimizer update',ylabel='Full-vocabulary answer-token loss');ax.legend();fig.tight_layout();fig.savefig(path/'loss_curves.png');plt.close(fig)
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('build','baseline','smoke','train','evaluate','compare','final-test'))
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--historical-manifest',type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    repository=Path(__file__).resolve().parents[1]
    if a.out.resolve()==repository or repository in a.out.resolve().parents:
        p.error('Generated manifests/results belong outside Git, normally on Google Drive')
    if a.out.resolve()==a.data_dir.resolve() or a.data_dir.resolve() in a.out.resolve().parents:
        p.error('Output must be outside the source dataset')
    if a.action=='build':
        manifest,mapping_bytes=generate(a.data_dir,a.historical_manifest)
        print(manifest['provenance']['method'])
        print(json.dumps({split:{k:v for k,v in report.items() if k!='image_reuse_counts'} for split,report in manifest['validation_report'].items()},indent=2))
        if not a.validate_only:
            if a.out.exists():p.error('Choose a new experiment root; build never overwrites existing output')
            path=a.out/'dataset';path.mkdir(parents=True)
            (path/'partitions.json').write_bytes(mapping_bytes)
            atomic_json(path/'manifest.json',manifest);atomic_json(path/'validation_report.json',manifest['validation_report'])
        return
    manifest=load_manifest(a.out,a.data_dir)
    if a.validate_only:
        print('Validated 800/100/100 images, 147/49/98 sets, all seven rules, balanced mappings and unique identifiability; no model loaded.');return
    if a.action=='baseline':evaluate(a.out,a.data_dir,manifest,resume=a.resume)
    elif a.action=='smoke':train(a.out,a.data_dir,manifest,resume=a.resume,smoke=True)
    elif a.action=='train':train(a.out,a.data_dir,manifest,resume=a.resume)
    elif a.action=='evaluate':evaluate(a.out,a.data_dir,manifest,'finetuned',resume=a.resume)
    elif a.action=='compare':print(json.dumps(compare_results(a.out,manifest,'validation'),indent=2))
    elif a.action=='final-test':
        selection=require_selection(a.out,manifest)
        # Validation comparison must already exist before locking the final test.
        compare_results(a.out,manifest,'validation')
        frozen=dict(selection=selection,manifest_hash=digest(manifest),model=MODEL,training=TRAIN,
                    adapter_identity=adapter_identity(a.out/'training/best_adapter'))
        freeze=a.out/'selection_freeze.json'
        if freeze.exists() and read(freeze)!=frozen:raise ValueError('Frozen selection changed')
        if not freeze.exists():atomic_json(freeze,frozen)
        for kind in ('pretrained','finetuned'):
            existing=(a.out/'final-test'/kind/'test').exists()
            evaluate(a.out,a.data_dir,manifest,kind,'test',a.resume and existing)
        print(json.dumps(compare_results(a.out,manifest,'test'),indent=2))


if __name__=='__main__':
    main()
