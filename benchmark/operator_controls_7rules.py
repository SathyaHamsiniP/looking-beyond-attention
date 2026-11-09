"""Experiment 07: recover and analyze A3; no model load by default."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import seven_rule_validation as validation
from seven_rule_validation import seven


def analyze(rows,view,historical=None):
    report=seven.metrics(list(rows.values()));groups=defaultdict(list)
    for row in rows.values():groups[row['set_id']].append(row['correct'])
    report['accuracy_95ci_set_cluster']=validation.bootstrap([sum(v)/len(v) for v in groups.values()])
    report['uncertainty']='5000 percentile resamples of image sets, seed 2026; accounts for rotations, not images reused across sets'
    report['recovery']='complete saved A3 predictions verified; zero model forwards'
    report['source_manifest_hash']=view['source_manifest_canonical_hash']
    if historical:
        history=seven.read(historical)
        report['historical_A1_A2']=history['conditions']
        report['comparison_caution']='A1/A2 are the paired original three generating rules with 3/7 choices. A3 uses seven generating rules and different validation sets; this is not a controlled causal effect of adding generating rules.'
    return report


def outputs(out,report,rows):
    out=validation.local_path(out);out.mkdir(parents=True,exist_ok=True)
    seven.atomic_json(out/'summary.json',report)
    if rows:
        with (out/'predictions.csv').open('w',newline='') as f:
            cols=('set_id','rotation','true_rule','predicted_rule','predicted_label','correct')
            w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows.values())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].bar(seven.LABELS,[report['answer_position_counts'][l] for l in seven.LABELS]);axes[0].set(xlabel='Predicted answer position',ylabel='Count')
        matrix=[[report['confusion'][r][p] for p in validation.choices.IDS] for r in validation.choices.IDS]
        axes[1].imshow(matrix);axes[1].set(xticks=range(7),yticks=range(7),xticklabels=validation.choices.IDS,yticklabels=validation.choices.IDS,xlabel='Predicted rule',ylabel='True rule')
        fig.tight_layout();fig.savefig(out/'answer_bias_and_confusion.png');plt.close(fig)
    text=('Seven generating rules, seven answer choices; validation only.\n'+json.dumps(report,indent=2)+'\n')
    (out/'summary.md').write_text(text);print(text,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--historical-a12',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--allow-inference',action='store_true',help='Explicitly permit up to 343 missing A3 forwards; never default')
    a=p.parse_args();out=validation.local_path(a.out)
    view=validation.validation_view(a.manifest,a.data_dir)
    if a.validate_only:
        rows,_,_=validation.verify_baseline(a.baseline,view)
        print(f'Validated 49 validation sets and {len(rows)} recovered predictions; zero GPU forwards.');return
    recovered=True
    try:rows,config,tokens=validation.verify_baseline(a.baseline,view)
    except FileNotFoundError:
        if not a.allow_inference:
            if (a.baseline/'metrics.json').exists():
                outputs(out,dict(status='historical aggregate only; no per-example predictions/CI available',historical_metrics=seven.read(a.baseline/'metrics.json')),None)
                return
            raise ValueError('A3 predictions unavailable; no model loaded. Recover sources or explicitly --allow-inference.')
        recovered=False
        import lora_7rules as model_helpers
        # Use the established evaluator with validation-only projection and local root.
        print('Explicit fallback workload: up to 343 pretrained validation forwards.',flush=True)
        out.mkdir(parents=True,exist_ok=True)
        model_helpers.evaluate(out,a.data_dir,seven.read(a.manifest),resume=(out/'evaluations/pretrained/validation').exists())
        # This newly produced projection has its own manifest identity, not an old-run claim.
        generated=out/'evaluations/pretrained/validation'
        rows=seven.load_rows(generated/'predictions.jsonl',validation.jobs(view));config=seven.read(generated/'config.json');tokens=seven.read(generated/'answer_tokens.json')
    print('A3 workload: '+('0 forwards (saved predictions verified).' if recovered else 'Fallback inference was explicitly requested; saved with reconstruction provenance.'),flush=True)
    report=analyze(rows,view,a.historical_a12)
    if not recovered:report['recovery']='explicit fallback inference on the verified source manifest'
    outputs(out,report,rows)
    seven.atomic_json(out/'validation_view.json',view)
    seven.atomic_json(out/'recovered_config.json',config)
    if tokens:seven.atomic_json(out/'answer_tokens.json',tokens)
    # Compact verified rows become the paired baseline cache for patching.
    path=out/'predictions.jsonl';content=''.join(json.dumps(rows[k])+'\n' for k in sorted(rows))
    if path.exists() and path.read_text()!=content:raise ValueError('Existing recovered prediction cache differs')
    path.write_text(content)


if __name__=='__main__':main()
