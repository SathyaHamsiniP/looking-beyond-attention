"""CPU checks of the original completed seven-rule LoRA records; no inference."""
import setup_paths
from benchmark_paths import BENCHMARK
from collections import Counter
import json
import math
from pathlib import Path
import unittest

import lora_7rules as seven

ROOT=BENCHMARK/'results/seven-rules/qwen3.5-4b'
LORA=ROOT/'lora'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class SavedLoRATests(unittest.TestCase):
    def test_original_file_hashes_and_exclusions(self):
        provenance=seven.read(LORA/'provenance.json')
        self.assertEqual(len(provenance['files']),12)
        expected={entry['path'] for entry in provenance['files']}|{'README.md','provenance.json'}
        self.assertEqual({str(p.relative_to(LORA)) for p in LORA.rglob('*') if p.is_file()},expected)
        for entry in provenance['files']:
            path=LORA/entry['path']
            self.assertEqual(path.stat().st_size,entry['bytes'])
            self.assertEqual(seven.file_hash(path),entry['sha256'])
        for path in LORA.rglob('*'):
            self.assertNotIn(path.suffix,('.npz','.pt','.safetensors','.zip'))
        self.assertEqual(seven.file_hash(ROOT/'operator-controls/predictions.jsonl'),provenance['baseline']['predictions_sha256'])
        self.assertEqual(seven.file_hash(ROOT/'operator-controls/recovered_config.json'),provenance['baseline']['config_sha256'])

    def test_successful_updates_and_epoch_selection(self):
        losses=seven.read(LORA/'training/losses.json');complete=seven.read(LORA/'training/complete.json')
        selection=seven.read(LORA/'training/selection.json');validations=seven.read(LORA/'training/validation_losses.json')
        self.assertEqual(len(losses),111);self.assertEqual(complete['successful_updates'],111)
        self.assertFalse(complete['smoke_test']);self.assertFalse(complete['test_used'])
        self.assertEqual(sum(r['examples'] for r in losses),complete['presentations']);self.assertEqual(complete['presentations'],441)
        self.assertEqual(sum(r['attempted_presentations'] for r in losses),complete['attempted_presentations'])
        self.assertEqual(Counter(r['epoch'] for r in losses),{1:37,2:37,3:37})
        for i,row in enumerate(losses,1):
            self.assertEqual(row['update'],i);self.assertTrue(row['optimizer_step_occurred']);self.assertTrue(row['finite_gradients'])
            self.assertEqual(row['nonfinite_gradient_values'],0);self.assertEqual(row['lora_b_matrices_changed'],248)
            self.assertTrue(math.isfinite(row['training_loss']))
            self.assertTrue(all(sid.startswith('train_') for sid in row['set_ids']))
            self.assertEqual(row['examples'],len(row['set_ids']))
        self.assertEqual([v['epoch'] for v in validations],[1,2,3]);self.assertTrue(all(v['sets']==49 for v in validations))
        best=min(validations,key=lambda v:v['loss'])
        self.assertEqual(best['epoch'],selection['epoch']);self.assertEqual(selection['epoch'],complete['selected_epoch']);self.assertEqual(selection['epoch'],3)
        self.assertEqual(best['loss'],selection['loss']);self.assertEqual(selection['loss'],1.5795268915137466)
        self.assertEqual(selection['criterion'],'lowest validation full-vocabulary answer-token loss')

    def test_matching_presentations_and_original_metrics(self):
        before=rows(ROOT/'operator-controls/predictions.jsonl');after=rows(LORA/'finetuned-validation/predictions.jsonl')
        self.assertEqual(len(before),343);self.assertEqual(len(after),343)
        self.assertEqual(sum(r['correct'] for r in before),48);self.assertEqual(sum(r['correct'] for r in after),112)
        b={r['key']:r for r in before};a={r['key']:r for r in after}
        self.assertEqual(len(a),343);self.assertEqual(set(a),set(b))
        fields=('key','set_id','true_rule','rotation','image_files','bits','option_mapping','prompt','true_label')
        summary=seven.read(LORA/'comparison/summary.json')
        for key,row in a.items():
            for field in fields:self.assertEqual(row[field],b[key][field],(key,field))
            expected={f:row[f] for f in fields}
            seven.comparison.validate_row(row,expected)
        for role,predictions in [('pretrained',before),('finetuned',after)]:
            computed=seven.metrics(predictions)
            for field,value in computed.items():
                if field!='caution':self.assertEqual(summary[role][field],value,(role,field))
        self.assertEqual(seven.read(LORA/'finetuned-validation/metrics.json'),summary['finetuned'])
        self.assertEqual(summary['accuracy_difference'],112/343-48/343)
        self.assertEqual(sum(b[k]['predicted_rule']!=a[k]['predicted_rule'] for k in a),summary['changed_predictions'])
        self.assertEqual(summary['changed_predictions'],311)

    def test_evaluation_records_match_selected_adapter(self):
        training=seven.read(LORA/'training/config.json');evaluation=seven.read(LORA/'finetuned-validation/config.json')
        baseline=seven.read(ROOT/'operator-controls/recovered_config.json');complete=seven.read(LORA/'training/complete.json')
        for field in ('manifest_hash','model','training','rules','prompts','source_hashes','software'):
            self.assertEqual(training[field],evaluation[field]);self.assertEqual(training[field],baseline[field])
        self.assertEqual(evaluation['kind'],'finetuned');self.assertEqual(evaluation['partition'],'validation');self.assertFalse(evaluation['smoke_test'])
        self.assertEqual(evaluation['adapter'],complete['adapter_identity'])
        self.assertEqual(set(evaluation['adapter']),{'adapter_config.json','adapter_model.safetensors','experiment.json'})
        self.assertTrue(all(len(sha)==64 for sha in evaluation['adapter'].values()))
        # These are matching saved identities, not independently checked weights.
        self.assertFalse(any(LORA.rglob('adapter_model.safetensors')))


if __name__=='__main__':unittest.main()
