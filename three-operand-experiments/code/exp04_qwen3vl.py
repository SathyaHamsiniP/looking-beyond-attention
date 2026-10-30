#!/usr/bin/env python3
"""Experiment 04: identify three-operand Boolean rules with Qwen3-VL-4B.

Converted from ``notebooks/exp04/Qwen3_VL_4B.ipynb``.
The experimental functions (sampling, prompts, scoring, statistics and
controls) are copied unchanged. This CLI replaces Colab-only setup and
avoids automatically running every experiment.

Examples (from the three-operand-experiments directory):
    python code/exp04_qwen3vl.py --validate-only
    python code/exp04_qwen3vl.py --upstream ../looking-beyond-attention-main --smoke-test
    python code/exp04_qwen3vl.py --upstream ../looking-beyond-attention-main --experiment main_72
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from itertools import product, permutations
from pathlib import Path

import numpy as np

# Populated during setup. The scientific functions below use these as they did
# in the notebook; no model is instantiated when importing this module.
MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
IMAGE_DIR = Path('data/images')
OUT = Path('runs/qwen3-vl-4b')
DATA_HASH = None
records = []
vlm = None
TOKEN_IDS = {}
LABELS = ('X', 'Y', 'Z')

# ---------------------------------------------------------------------------
# ORIGINAL EXPERIMENT FUNCTIONS — copied without changing their computations
# ---------------------------------------------------------------------------

EXPRESSIONS = {'R1': '(A AND B) OR C', 'R2': '(A AND C) OR B', 'R3': '(B AND C) OR A'}
RULES = tuple(EXPRESSIONS)
POOL = [r for r in records if r['variation'] < 4]  # hold out variations 4–5
N_IMAGES, SETS_PER_RULE, SEED = 10, 8, 2026

def satisfies(rule, bits):
    a,b,c = map(bool,bits)
    return {'R1': (a and b) or c, 'R2': (a and c) or b,
            'R3': (b and c) or a}[rule]

def compatible_rules(items):
    return [r for r in RULES if all(satisfies(r, x['bits']) for x in items)]

def sample_set(rule, idx, mode='coverage', n=N_IMAGES, seed=SEED):
    """coverage: all 5 positive types; iid: positive draws with replacement;
       decisive_once: only one one-hot positive scene (for ablation)."""
    assert mode in ('coverage', 'iid', 'decisive_once')
    rng = random.Random(seed + {'coverage':0,'iid':100000,'decisive_once':200000}[mode]
                        + 1000*RULES.index(rule) + idx)
    admissible = [b for b in product((0,1), repeat=3) if satisfies(rule,b)]
    eligible = [r for r in POOL if satisfies(rule,r['bits'])]
    if mode == 'iid':
        chosen = [rng.choice(eligible) for _ in range(n)]
    else:
        assert n >= len(admissible)
        chosen = [rng.choice([r for r in eligible if tuple(r['bits']) == b])
                  for b in admissible]
        unique = next(b for b in admissible if sum(b) == 1)
        remainder = [r for r in eligible if r['filename'] not in {x['filename'] for x in chosen}
                     and (mode != 'decisive_once' or tuple(r['bits']) != unique)]
        chosen += rng.sample(remainder, n-len(chosen))
    rng.shuffle(chosen)
    consistent = compatible_rules(chosen)
    if mode != 'iid':
        assert consistent == [rule]
    if mode == 'decisive_once':
        assert sum(sum(x['bits'])==1 for x in chosen) == 1
    return dict(group_id=f'{mode}:{rule}:{idx}', hidden_rule=rule, set_index=idx,
                sampling=mode, items=chosen, scene_bits=[x['bits'] for x in chosen],
                image_files=[x['filename'] for x in chosen],
                distinct_types_shown=len({tuple(x['bits']) for x in chosen}),
                admissible_types=len(admissible), consistent_candidates=consistent,
                identifiable=(consistent==[rule]))

def layouts(control='cyclic'):
    fixed = dict(zip(LABELS,RULES))
    if control == 'fixed': return [(fixed, LABELS)]
    if control == 'cyclic':
        return [(dict(zip(LABELS,p)), LABELS) for p in
                [('R1','R2','R3'),('R2','R3','R1'),('R3','R1','R2')]]
    if control == 'position':  # fixed label -> rule; change physical line order
        return [(fixed,p) for p in permutations(LABELS)]
    if control == 'relabel':   # fixed expression positions; change X/Y/Z assignments
        return [(dict(zip(p,RULES)),p) for p in permutations(LABELS)]
    raise ValueError(control)



def scene_text(scene):
    return '\n'.join(f"Scene {i}: A={b[0]}, B={b[1]}, C={b[2]}"
                     for i,b in enumerate(scene['scene_bits'],1))

def build_input(scene, mapping, display, modality='images'):
    """Main image prompt matches the earlier notebook's 72-trial wording."""
    assert modality in ('images','text','captioned')
    choices = '\n'.join(f'{label}: {EXPRESSIONS[mapping[label]]}' for label in display)
    prompt = f"""Images 1-{len(scene["image_files"])} all satisfy ONE hidden Boolean rule.

A is true when a green cube is present.
B is true when a blue sphere is present.
C is true when a red cylinder is present.
Each concept requires the exact colour AND shape.
Any other object is an irrelevant distractor.

Candidate rules:
{choices}

AND requires both inputs to be true.
OR requires at least one input to be true, including both.
Follow the parentheses.

Pick the rule consistent with every image.
Reply with nothing but its label: X, Y, or Z.
ANSWER:"""
    if modality != 'images':
        prompt = ('Ground-truth Boolean scene descriptions (1 = present, 0 = absent):\n'
                  + scene_text(scene) + '\n\n' + prompt.replace('Images 1-', 'Scenes 1-'))
    paths = ([] if modality == 'text' else
             [str(IMAGE_DIR/name) for name in scene['image_files']])
    return paths, prompt

def score_trial(scene, mapping, display, modality='images'):
    paths,prompt = build_input(scene,mapping,display,modality)
    raw = option_probs(vlm,paths,prompt,TOKEN_IDS)  # original Roshan readout
    probs = {label:float(raw[label]) for label in LABELS}
    s = sum(probs.values())
    assert s > 0 and all(math.isfinite(x) and x>=0 for x in probs.values()), probs
    probs = {k:v/s for k,v in probs.items()}
    selected = max(LABELS,key=lambda l:probs[l])
    true_label = next(l for l in LABELS if mapping[l]==scene['hidden_rule'])
    sorted_p = sorted(probs.values(),reverse=True)
    return dict(prompt=prompt, label_mapping=dict(mapping), display_order=list(display),
                modality=modality, probabilities=probs, chosen_label=selected,
                chosen_rule=mapping[selected], true_label=true_label,
                chosen_position=list(display).index(selected)+1,
                correct=(mapping[selected]==scene['hidden_rule']),
                true_probability=probs[true_label], confidence=probs[selected],
                margin=sorted_p[0]-sorted_p[1],
                nll=-math.log(max(probs[true_label],1e-12)),
                brier=sum((probs[l]-float(l==true_label))**2 for l in LABELS),
                trial_entropy_bits=-sum(p*math.log2(p) for p in probs.values() if p>0))

def entropy(values):
    ct=Counter(values); n=len(values)
    return -sum((k/n)*math.log2(k/n) for k in ct.values() if k)

def mutual_information(truth,pred):
    n=len(truth); a=Counter(truth); b=Counter(pred); ab=Counter(zip(truth,pred))
    return sum((count/n)*math.log2((count*n)/(a[x]*b[y]))
               for (x,y),count in ab.items())

def analyse(rows, run_dir=None, permutations_n=1000, bootstraps=1000):
    assert rows
    truth=[r['hidden_rule'] for r in rows]
    pred=[r['chosen_rule'] for r in rows]
    groups=defaultdict(list)
    for r in rows: groups[r['group_id']].append(r)
    keys=sorted(groups)
    rng=np.random.default_rng(2026)
    observed_mi=mutual_information(truth,pred)
    null=[]
    group_truth=[groups[k][0]['hidden_rule'] for k in keys]
    for _ in range(permutations_n):
        labels=dict(zip(keys,rng.permutation(group_truth)))
        null.append(mutual_information([labels[r['group_id']] for r in rows],pred))
    boot=[]
    for _ in range(bootstraps):
        sampled=rng.choice(keys,size=len(keys),replace=True)
        sample=[r for k in sampled for r in groups[k]]
        boot.append(np.mean([r['correct'] for r in sample]))
    identifiable=[r for r in rows if r['identifiable']]
    summary=dict(trials=len(rows), independent_scene_sets=len(keys),
                 accuracy=float(np.mean([r['correct'] for r in rows])),
                 accuracy_95ci_cluster=[float(x) for x in np.percentile(boot,[2.5,97.5])],
                 identifiable_fraction=len(identifiable)/len(rows),
                 identifiable_accuracy=(float(np.mean([r['correct'] for r in identifiable]))
                                        if identifiable else None),
                 chance_baseline=1/len(RULES),
                 output_entropy_bits=entropy(pred),
                 mutual_information_bits=observed_mi,
                 mi_null_mean=float(np.mean(null)),
                 mi_minus_null=float(observed_mi-np.mean(null)),
                 mi_permutation_p=(1+sum(x>=observed_mi for x in null))/(1+len(null)),
                 mean_true_prob=float(np.mean([r['true_probability'] for r in rows])),
                 mean_nll=float(np.mean([r['nll'] for r in rows])),
                 mean_brier=float(np.mean([r['brier'] for r in rows])),
                 chosen_labels=dict(Counter(r['chosen_label'] for r in rows)),
                 chosen_positions=dict(Counter(str(r['chosen_position']) for r in rows)),
                 chosen_rules=dict(Counter(pred)),
                 per_rule={r:dict(n=sum(x['hidden_rule']==r for x in rows),
                                  accuracy=float(np.mean([x['correct'] for x in rows if x['hidden_rule']==r])))
                           for r in RULES},
                 confusion={r:{p:sum(x['chosen_rule']==p for x in rows if x['hidden_rule']==r)
                                for p in RULES} for r in RULES},
                 by_diversity={str(d):{'n':sum(x['distinct_types_shown']==d for x in rows),
                                       'accuracy':float(np.mean([x['correct'] for x in rows
                                                                 if x['distinct_types_shown']==d]))}
                               for d in sorted({x['distinct_types_shown'] for x in rows})})
    if run_dir is not None:
        (Path(run_dir)/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f"Accuracy: {summary['accuracy']:.1%} | 95% set-bootstrap CI: {summary['accuracy_95ci_cluster']}")
    print(f"MI: {observed_mi:.3f} bits | null mean: {summary['mi_null_mean']:.3f} | p: {summary['mi_permutation_p']:.4f}")
    print('Confusion matrix:',summary['confusion'])
    print('Labels:',summary['chosen_labels'],'| Positions:',summary['chosen_positions'])
    return summary

def trial_status_line(run_name, completed, total, row):
    """One readable progress line; original trial IDs remain in runs.jsonl."""
    names = {
        'main_72': 'Main experiment', 'text_only': 'Text-only',
        'captioned': 'Captioned', 'position_order': 'Option order',
        'label_assignment': 'Label assignment', 'iid_positive': 'IID sampling',
        'ablation': 'Ablation', 'discovery_open': 'Open discovery',
        'discovery_guided': 'Guided discovery', 'perception': 'Object recognition',
        'gates': 'Boolean gates',
    }
    name = names.get(run_name, run_name.replace('_', ' ').capitalize())
    prefix = f'{name} {completed}/{total}'
    actual = row.get('hidden_rule')
    predicted = row.get('chosen_rule', row.get('predicted_rule'))
    if actual and predicted is not None:
        verdict = 'Correct' if row.get('correct') else 'Incorrect'
        choice = (f" | Choice {row['chosen_label']} (position {row['chosen_position']})"
                  if 'chosen_label' in row and 'chosen_position' in row else '')
        return f'{prefix} | Actual: {actual} | Predicted: {predicted} | {verdict}{choice}'
    if run_name.startswith('discovery_'):
        return f'{prefix} | Actual: {actual} | No final rule parsed'
    if run_name == 'ablation':
        verdict = 'Correct' if row.get('full_correct') else 'Incorrect'
        return f"{prefix} | Actual: {actual} | Full set: {verdict} | Decisive drop: {row['decisive_drop']:+.3f}"
    if run_name == 'perception':
        return f"{prefix} | {row['filename']} | {'Correct' if row['correct'] else 'Incorrect'}"
    if run_name == 'gates':
        d = 'correct' if row['d_correct'] else 'incorrect'
        y = 'correct' if row['y_correct'] else 'incorrect'
        return f'{prefix} | Rule: {actual} | Intermediate: {d} | Output: {y}'
    return f'{prefix} | Saved'

def saved_runner(run_name, config, jobs, infer):
    """Generic resumable JSONL runner. Exactly one append per completed inference unit."""
    folder=OUT/run_name; folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/'config.json'; rows_path=folder/'runs.jsonl'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, f'Config changed: use another run_name: {run_name}'
    else:
        config_path.write_text(json.dumps(config,indent=2))
    rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()] if rows_path.exists() else []
    done=[r['trial_id'] for r in rows]
    assert len(done)==len(set(done)), f'Duplicate trial ID in {rows_path}'
    seen=set(done)
    for trial_id, payload in jobs:
        if trial_id in seen: continue
        result=infer(payload)
        row={'trial_id':trial_id, **result}
        with rows_path.open('a') as f:
            f.write(json.dumps(row)+'\n')
        rows.append(row); seen.add(trial_id)
        print(trial_status_line(run_name, len(rows), len(jobs), row), flush=True)
    print(f'Completed {len(rows)}/{len(jobs)} trials | Results: {rows_path.parent}')
    return rows,folder

def run_experiment(name='main_72', sampling='coverage', control='cyclic',
                   modality='images', n_sets=SETS_PER_RULE, n_images=N_IMAGES, seed=SEED):
    plan=list(layouts(control))
    config={'version':'exp8-three-operand-v1','model':MODEL_ID,'data_hash':DATA_HASH,
            'sampling':sampling,'control':control,'modality':modality,'sets_per_rule':n_sets,
            'images_per_set':n_images,'seed':seed,'readout':'Roshan option_probs X/Y/Z',
            'expressions':EXPRESSIONS,'labels':list(LABELS),'token_ids':dict(TOKEN_IDS)}
    jobs=[]
    for rule in RULES:
        for i in range(n_sets):
            scene=sample_set(rule,i,sampling,n_images,seed)
            for k,(mapping,display) in enumerate(plan):
                jobs.append((f'{scene["group_id"]}:{control}:{k}:{modality}',
                             (scene,mapping,display)))
    def infer(payload):
        scene,mapping,display=payload
        return dict(group_id=scene['group_id'], hidden_rule=scene['hidden_rule'],
                    set_index=scene['set_index'], sampling=scene['sampling'],
                    image_files=scene['image_files'], scene_bits=scene['scene_bits'],
                    distinct_types_shown=scene['distinct_types_shown'],
                    admissible_types=scene['admissible_types'],
                    identifiable=scene['identifiable'],
                    consistent_candidates=scene['consistent_candidates'],
                    **score_trial(scene,mapping,display,modality))
    rows,folder=saved_runner(name,config,jobs,infer)
    return rows,analyse(rows,folder),folder

def drop_scene(scene, j):
    selected=[x for k,x in enumerate(scene['items']) if k!=j]
    return dict(scene, items=selected, scene_bits=[x['bits'] for x in selected],
                image_files=[x['filename'] for x in selected],
                consistent_candidates=compatible_rules(selected),
                identifiable=compatible_rules(selected)==[scene['hidden_rule']])

def run_ablation(n_sets=SETS_PER_RULE):
    mapping,display=layouts('fixed')[0]
    jobs=[]
    for rule in RULES:
        for i in range(n_sets):
            scene=sample_set(rule,i,'decisive_once')
            jobs.append((scene['group_id'],scene))
    cfg={'version':1,'model':MODEL_ID,'data_hash':DATA_HASH,'n_sets':n_sets,'seed':SEED}
    def infer(scene):
        bits=scene['scene_bits']; rule=scene['hidden_rule']
        decisive=next(i for i,b in enumerate(bits) if sum(b)==1)
        control=next(i for i,b in enumerate(bits) if sum(b)>1)
        minus_dec=drop_scene(scene,decisive); minus_ctl=drop_scene(scene,control)
        assert set(minus_dec['consistent_candidates'])==set(RULES)
        assert minus_ctl['consistent_candidates']==[rule]
        full=score_trial(scene,mapping,display)
        dec=score_trial(minus_dec,mapping,display)
        ctl=score_trial(minus_ctl,mapping,display)
        return dict(group_id=scene['group_id'],hidden_rule=rule,scene_bits=bits,
                    decisive_index=decisive+1, control_index=control+1,
                    full=full['probabilities'],minus_decisive=dec['probabilities'],
                    minus_control=ctl['probabilities'],
                    full_correct=full['correct'],
                    decisive_drop=full['true_probability']-dec['true_probability'],
                    control_drop=full['true_probability']-ctl['true_probability'])
    rows,folder=saved_runner('ablation',cfg,jobs,infer)
    summary={'n':len(rows),'full_accuracy':float(np.mean([r['full_correct'] for r in rows])),
             'mean_decisive_drop':float(np.mean([r['decisive_drop'] for r in rows])),
             'mean_control_drop':float(np.mean([r['control_drop'] for r in rows])),
             'mean_paired_effect':float(np.mean([r['decisive_drop']-r['control_drop'] for r in rows]))}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)
    return rows,summary

def parse_rule(reply):
    match=re.search(r'\(\s*([ABC])\s+AND\s+([ABC])\s*\)\s+OR\s+([ABC])\b',reply,re.I)
    if match and set(x.upper() for x in match.groups())==set('ABC'):
        return {'C':'R1','B':'R2','A':'R3'}[match.group(3).upper()]
    return None

def discovery_prompt(scene, mode):
    head=f"""You are given ten Boolean scenes (1 = TRUE, 0 = FALSE).
{scene_text(scene)}

All scenes satisfy one hidden rule of the form (P AND Q) OR R,
where P, Q, R are A, B, C in some order, used once each.
AND means both true; OR means at least one true.
"""
    if mode=='open':
        return head+'Return ONLY the completed expression: (<variable> AND <variable>) OR <variable>\nANSWER:'
    return head+"""Find a scene with exactly one TRUE variable. That variable is R,
outside AND. Write exactly three lines:
Decisive scene: <number>
Outside variable: <A/B/C>
Expression: (<variable> AND <variable>) OR <variable>"""

def run_discovery(mode='open',n_sets=SETS_PER_RULE):
    assert mode in ('open','guided')
    cfg={'version':1,'model':MODEL_ID,'data_hash':DATA_HASH,'mode':mode,
         'sets_per_rule':n_sets,'seed':SEED,'modality':'text','sampling':'coverage'}
    jobs=[]
    for rule in RULES:
        for i in range(n_sets):
            scene=sample_set(rule,i)
            jobs.append((scene['group_id'],scene))
    def infer(scene):
        prompt=discovery_prompt(scene,mode)
        reply=str(vlm.chat([],prompt,max_new_tokens=150,temperature=0.0)).strip()
        predicted=parse_rule(reply)
        row=dict(group_id=scene['group_id'],hidden_rule=scene['hidden_rule'],
                 scene_bits=scene['scene_bits'],prompt=prompt,response=reply,
                 predicted_rule=predicted,correct=predicted==scene['hidden_rule'])
        if mode=='guided':
            one_hot=[i for i,b in enumerate(scene['scene_bits'],1) if sum(b)==1]
            expected={'R1':'C','R2':'B','R3':'A'}[scene['hidden_rule']]
            pos=re.search(r'Decisive scene:\s*(?:Scene\s*)?(\d+)',reply,re.I)
            outer=re.search(r'Outside variable:\s*([ABC])\b',reply,re.I)
            pos_val=int(pos.group(1)) if pos else None
            outer_val=outer.group(1).upper() if outer else None
            row.update(decisive_scene=pos_val,valid_decisive_scenes=one_hot,
                       scene_correct=pos_val in one_hot, expected_outer=expected,
                       predicted_outer=outer_val,outer_correct=outer_val==expected,
                       internally_consistent=(outer_val is not None and predicted is not None
                                              and outer_val=={'R1':'C','R2':'B','R3':'A'}[predicted]))
        return row
    rows,folder=saved_runner('discovery_'+mode,cfg,jobs,infer)
    summary={'n':len(rows),'rule_accuracy':float(np.mean([r['correct'] for r in rows])),
             'prediction_counts':dict(Counter(str(r['predicted_rule']) for r in rows))}
    if mode=='guided':
        summary.update({k+'_rate':float(np.mean([r[k] for r in rows])) for k in
                        ('scene_correct','outer_correct','internally_consistent')})
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)
    return rows,summary

def run_perception():
    """Eight held-out images; parse A/B/C. Note: only tests one variation per type."""
    jobs=[(str(r['filename']),r) for r in records if r['variation']==4]
    cfg={'version':1,'model':MODEL_ID,'data_hash':DATA_HASH,'modality':'images','variation':4}
    prompt="""Inspect this image. A = green cube, B = blue sphere, C = red cylinder.
Each means exact colour AND shape. Ignore distractors.
Reply with three binary values in format A=0, B=0, C=0 (replace as needed)."""
    def infer(r):
        reply=str(vlm.chat([str(IMAGE_DIR/r['filename'])],prompt,
                           max_new_tokens=60,temperature=0.0))
        found=[re.search(rf'\b{v}\s*[:=]\s*([01])',reply,re.I) for v in 'ABC']
        pred=[int(m.group(1)) if m else None for m in found]
        return dict(group_id=r['filename'],hidden_rule=None,filename=r['filename'],
                    ground_truth=r['bits'],prediction=pred,response=reply,
                    correct=pred==r['bits'])
    rows,folder=saved_runner('perception',cfg,jobs,infer)
    summary={'n':len(rows),'accuracy':float(np.mean([r['correct'] for r in rows]))}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)
    return rows,summary

def run_gates():
    """3 Boolean rules x 8 inputs; compare intermediate D and final Y."""
    indices={'R1':(0,1,2),'R2':(0,2,1),'R3':(1,2,0)}
    jobs=[(f'{rule}:{"".join(map(str,bits))}',(rule,bits))
          for rule in RULES for bits in product((0,1),repeat=3)]
    cfg={'version':1,'model':MODEL_ID,'task':'two_gate_text','rule_order':list(RULES)}
    def infer(pair):
        rule,bits=pair; i,j,k=indices[rule]; symbols='ABC'
        prompt=f"""A={bits[0]}, B={bits[1]}, C={bits[2]}.
D = {symbols[i]} AND {symbols[j]}
Y = D OR {symbols[k]}
Use Boolean gates. Reply exactly on two lines: D=<0/1> and Y=<0/1>."""
        response=str(vlm.chat([],prompt,max_new_tokens=50,temperature=0.0))
        def read(v):
            m=re.search(rf'\b{v}\s*[:=]\s*([01])',response,re.I)
            return int(m.group(1)) if m else None
        d,y=read('D'),read('Y'); d_true=bits[i]&bits[j]; y_true=int(satisfies(rule,bits))
        return dict(group_id=f'{rule}:{bits}',hidden_rule=rule,bits=list(bits),
                    predicted_d=d,predicted_y=y,expected_d=d_true,expected_y=y_true,
                    d_correct=d==d_true,y_correct=y==y_true,response=response)
    rows,folder=saved_runner('gates',cfg,jobs,infer)
    summary={'n':len(rows),'d_accuracy':float(np.mean([r['d_correct'] for r in rows])),
             'y_accuracy':float(np.mean([r['y_correct'] for r in rows])),
             'both_correct':float(np.mean([r['d_correct'] and r['y_correct'] for r in rows]))}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)
    return rows,summary


# ---------------------------------------------------------------------------
# REPRODUCIBLE CLI — notebook setup and manually selected run cells replaced
# ---------------------------------------------------------------------------

EXPERIMENTS = (
    'main_72', 'text_only', 'captioned', 'position_order',
    'label_assignment', 'iid_positive', 'ablation', 'discovery_open',
    'discovery_guided', 'perception', 'gates',
)


def resolve_upstream_code(upstream: Path) -> Path:
    """Return a checkout's `code/` directory containing Roshan's loader.

    Accept the upstream repository root, its `code/` directory, or the parent
    of an extracted `looking-beyond-attention-main/` checkout.
    """
    upstream = upstream.expanduser().resolve()
    candidates = [upstream, upstream / 'code']
    if upstream.is_dir():
        candidates.extend(p / 'code' for p in upstream.glob('looking-beyond-attention*') if p.is_dir())
    for code_dir in candidates:
        if ((code_dir / 'vlm' / 'runner.py').is_file()
                and (code_dir / 'experiments' / 'readout.py').is_file()):
            return code_dir
    raise FileNotFoundError(
        f"Could not find vlm/runner.py and experiments/readout.py below {upstream}. "
        "Pass --upstream PATH_TO_ORIGINAL_REPO or --upstream PATH_TO_CODE_DIR."
    )


def configure_data(data_dir: Path, out_dir: Path) -> None:
    """Load the 48 existing scenes; never regenerate or alter the dataset."""
    global records, IMAGE_DIR, OUT, DATA_HASH, POOL
    data_dir = data_dir.expanduser().resolve()
    metadata_path = data_dir / 'metadata.json'
    IMAGE_DIR = data_dir / 'images'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Missing scene metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    records = metadata['records']
    POOL = [r for r in records if r['variation'] < 4]
    if len(records) != 48:
        raise ValueError(f'Expected 48 scene records, found {len(records)}')
    missing = [r['filename'] for r in records if not (IMAGE_DIR / r['filename']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} scene images; first: {missing[:3]}')
    DATA_HASH = hashlib.sha256(metadata_path.read_bytes()).hexdigest()[:16]
    OUT = out_dir.expanduser().resolve()


def verify_sampling() -> None:
    """CPU-only preflight: check samples and all three generating rules."""
    seen = set()
    for rule in RULES:
        for idx in range(SETS_PER_RULE):
            scene = sample_set(rule, idx)
            assert scene['identifiable'] and len(scene['image_files']) == N_IMAGES
            assert scene['consistent_candidates'] == [rule]
            seen.add((rule, idx))
    print(f'Dataset OK: {len(records)} images; {len(seen)} rule/scene sets; seed={SEED}')
    print(f'Output location: {OUT}')


def load_upstream_model(code_dir: Path, model_id: str, precision: str,
                        device: str, max_pixels: int) -> None:
    """Use Roshan's *original* Qwen3-VL wrapper and next-token readout."""
    global vlm, MODEL_ID, TOKEN_IDS, option_probs
    import torch
    if precision == '4bit' and not torch.cuda.is_available():
        raise RuntimeError(
            '4-bit Qwen inference requires a CUDA-capable GPU in this setup. '
            'Use a Colab T4 runtime, or choose an appropriate supported precision/device.'
        )
    sys.path.insert(0, str(code_dir))
    from vlm.runner import load_model
    from experiments.readout import tokenizer_of, option_token_ids
    from experiments.readout import option_probs as upstream_option_probs

    MODEL_ID = model_id
    option_probs = upstream_option_probs
    vlm = load_model(MODEL_ID, precision=precision, device=device, max_pixels=max_pixels)
    TOKEN_IDS = option_token_ids(tokenizer_of(vlm), list(LABELS))
    assert set(TOKEN_IDS) == set(LABELS), TOKEN_IDS
    assert len(set(TOKEN_IDS.values())) == 3, 'Duplicate answer-label token IDs'
    print('Model:', vlm.info(), '| Answer token IDs:', TOKEN_IDS)


def smoke_test() -> None:
    """One image inference, without saving results or running the 72 trials."""
    scene = sample_set('R1', 0)
    mapping, display = layouts('fixed')[0]
    probe = score_trial(scene, mapping, display, 'images')
    print('Smoke test | Actual: R1 | Predicted:', probe['chosen_rule'],
          '| Choice:', probe['chosen_label'], '| P(X/Y/Z):', probe['probabilities'])
    print('The smoke test checks execution, NOT rule-identification accuracy.')


def run_selected(name: str) -> None:
    """Use the notebook's exact experiment functions with its original defaults."""
    if name == 'main_72':
        run_experiment(name='main_72', sampling='coverage', control='cyclic', modality='images')
    elif name == 'text_only':
        run_experiment('text_only', control='fixed', modality='text')
    elif name == 'captioned':
        run_experiment('captioned', control='fixed', modality='captioned')
    elif name == 'position_order':
        run_experiment('position_order', control='position')
    elif name == 'label_assignment':
        run_experiment('label_assignment', control='relabel')
    elif name == 'iid_positive':
        run_experiment('iid_positive', sampling='iid', control='fixed')
    elif name == 'ablation':
        run_ablation()
    elif name == 'discovery_open':
        run_discovery('open')
    elif name == 'discovery_guided':
        run_discovery('guided')
    elif name == 'perception':
        run_perception()
    elif name == 'gates':
        run_gates()
    else:
        raise ValueError(f'Unknown experiment: {name}')


def make_parser() -> argparse.ArgumentParser:
    script_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description='Run Qwen3-VL three-operand Boolean rule experiments (Exp 04).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=Path, default=script_root / 'data',
                        help='Directory containing metadata.json and images/')
    parser.add_argument('--out', type=Path, default=script_root / 'runs' / 'qwen3-vl-4b',
                        help='Directory for NEW results; existing archive is never overwritten by default')
    parser.add_argument('--upstream', type=Path,
                        help="Path to Roshan's repository root, its code/ directory, or its ZIP archive")
    parser.add_argument('--experiment', choices=EXPERIMENTS, default='main_72',
                        help='Run exactly one experiment; no others are triggered')
    parser.add_argument('--validate-only', action='store_true',
                        help='Check dataset and sampling; do not load any model')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run one image-trial inference only; do not save results')
    parser.add_argument('--model', default=MODEL_ID, help='Hugging Face model ID')
    parser.add_argument('--precision', choices=('4bit', 'fp16', 'bf16', 'fp32'), default='4bit')
    parser.add_argument('--device', default='auto', help="Passed to Roshan's load_model")
    parser.add_argument('--max-pixels', type=int, default=384 * 28 * 28,
                        help='Maximum image pixels as used in the notebook')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    configure_data(args.data_dir, args.out)
    verify_sampling()
    if args.validate_only:
        print('CPU-only validation passed; no GPU or model inference was used.')
        return 0
    if args.upstream is None:
        raise SystemExit('Missing --upstream PATH_TO_ROSHAN_REPOSITORY (see README).')
    # Accept a checkout or a ZIP of the upstream repository. Extraction uses a
    # short-lived directory; no upstream code is redistributed with our script.
    with tempfile.TemporaryDirectory(prefix='exp04_qwen3_upstream_') as tmp:
        upstream = args.upstream.expanduser().resolve()
        if upstream.is_file() and upstream.suffix.lower() == '.zip':
            with zipfile.ZipFile(upstream) as archive:
                for member in archive.namelist():
                    parts = Path(member).parts
                    if member.startswith('/') or '..' in parts:
                        raise ValueError(f'Unsafe path in upstream ZIP: {member}')
                archive.extractall(tmp)
            upstream = Path(tmp)
        code_dir = resolve_upstream_code(upstream)
        load_upstream_model(code_dir, args.model, args.precision, args.device, args.max_pixels)
        if args.smoke_test:
            smoke_test()
        else:
            print(f'Running {args.experiment}; completed trials resume from {OUT / args.experiment}')
            run_selected(args.experiment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
