"""CPU checks of saved result files and measurements."""
import setup_paths
from benchmark_paths import BENCHMARK, source_path
import csv
import json
from pathlib import Path
import unittest

import activation_patching_7rules as patching
import lora_7rules as seven

ROOT=BENCHMARK/'results'


class SavedResultTests(unittest.TestCase):
    def test_import_receipts_match_all_original_files(self):
        receipts=list(ROOT.rglob('provenance.json'))
        self.assertGreaterEqual(len(receipts),6)
        for provenance in receipts:
            for entry in seven.read(provenance)['files']:
                path=provenance.parent/entry['path']
                self.assertEqual(path.stat().st_size,entry['bytes'],str(path))
                self.assertEqual(seven.file_hash(path),entry['sha256'],str(path))
        self.assertFalse(any(p.suffix in ('.npz','.pt','.safetensors') for p in ROOT.rglob('*')))

    def test_seven_rule_controls_and_every_patching_effect(self):
        base=ROOT/'seven-rules/qwen3.5-4b'
        rows=[json.loads(line) for line in (base/'operator-controls/predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),343);self.assertEqual(sum(r['correct'] for r in rows),48)
        summary=seven.read(base/'operator-controls/summary.json');computed=seven.metrics(rows)
        for key in ('accuracy','per_rule','confusion','answer_position_counts','rotation_consistency'):
            self.assertEqual(computed[key],summary[key],key)
        folder=base/'activation-patching';cfg=seven.read(folder/'config.json')
        pairs={p['pair_id']:p for p in cfg['pairs']}
        rows=[json.loads(line) for line in (folder/'patches.jsonl').read_text().splitlines()]
        expected={f"{p}:{rotation}:{layer}:{position}" for p in pairs for rotation in cfg['rotations'] for layer in cfg['layers'] for position in cfg['positions']}
        self.assertEqual(len(rows),896);self.assertEqual({r['key'] for r in rows},expected)
        for row in rows:
            actual=patching.effect(pairs[row['pair_id']],row['rotation'],row['layer'],row['position'],row['before_probabilities'],row['donor_probabilities'],row['patched_probabilities'])
            for key,value in actual.items():self.assertEqual(row[key],value,(row['key'],key))
        with (folder/'layer_effects.csv').open() as stream:aggregates=list(csv.DictReader(stream))
        for aggregate in aggregates:
            group=[r for r in rows if r['condition']==aggregate['condition'] and r['position']==aggregate['position'] and r['layer']==int(aggregate['layer'])]
            for key in ('delta_true_rule','absolute_delta_true_rule','delta_donor_answer','total_variation','top_rule_changed'):
                self.assertAlmostEqual(sum(r[key] for r in group)/len(group),float(aggregate[key]),places=12)

    def test_three_rule_probe_and_lora_predictions(self):
        base=ROOT/'three-rules/qwen3.5-4b'
        with (base/'probing/probe_predictions.csv').open() as stream:predictions=list(csv.DictReader(stream))
        with (base/'probing/layer_accuracy.csv').open() as stream:accuracies=list(csv.DictReader(stream))
        self.assertEqual(len(predictions),3840);self.assertEqual(len(accuracies),128)
        for row in accuracies:
            subset=[p for p in predictions if all(p[k]==row[k] for k in ('images','layer','control'))]
            self.assertEqual(len(subset),30)
            self.assertAlmostEqual(sum(int(p['correct']) for p in subset)/30,float(row['accuracy']))
        for role,correct in [('pretrained',29),('finetuned',30)]:
            rows=[json.loads(line) for line in (base/f'lora/{role}/predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),90);self.assertEqual(sum(r['correct'] for r in rows),correct)
        self.assertEqual(seven.read(base/'lora/training/complete.json')['optimizer_updates'],46)


if __name__=='__main__':unittest.main()
