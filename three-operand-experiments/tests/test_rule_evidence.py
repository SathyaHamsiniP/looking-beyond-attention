"""Logical design checks and mocked pilot orchestration, without GPU inference."""
import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'code/02-probing'))
import rule_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        records = json.loads((PROJECT / 'data/metadata.json').read_text())['records']
        self.families = evidence.make_dataset(records)

    def test_pairs_labels_and_balancing(self):
        evidence.validate(self.families)
        self.assertEqual(len(self.families), 9)
        table = evidence.baseline_table(self.families)
        for family in self.families:
            for job in evidence.presentations(family):
                baselines = evidence.baseline_probs(job, table)
                for probabilities in baselines.values():
                    self.assertAlmostEqual(sum(probabilities.values()), 1)
                    expected = 1 if job['condition'] == 'both' else 0.5
                    self.assertAlmostEqual(probabilities[family['rule']], expected)
                prompt = evidence.prompt_for(job)
                self.assertEqual(prompt.count(': positive'), 1)
                self.assertEqual(prompt.count(': negative'), 3)
                self.assertNotIn('R1', prompt)
        report = evidence.report(self.families)
        self.assertEqual(report['test']['unique_images'], 6)
        self.assertEqual(report['train']['unique_images'], 12)
        self.assertEqual(report['test']['max_mutually_image_disjoint_families'], 1)

    def test_incorrect_label_is_rejected(self):
        self.families[0]['conditions']['both'][0]['positive'] = True
        with self.assertRaises(AssertionError):
            evidence.validate(self.families)

    def test_repeated_presentations_are_grouped(self):
        table = evidence.baseline_table(self.families)
        rows = []
        for family in self.families:
            if family['split'] != 'test':
                continue
            for job in evidence.presentations(family):
                for method, probabilities in evidence.baseline_probs(job, table).items():
                    row = dict(method=method, family_id=family['family_id'], condition=job['condition'],
                               truth=family['rule'], selection_accuracy=probabilities[family['rule']],
                               consistent_probability=1)
                    row.update({f'probability_{r}': probabilities[r] for r in evidence.experiment.RULES})
                    rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            evidence.summarize(rows, Path(folder))
            with (Path(folder) / 'family_summary.csv').open() as f:
                grouped = list(csv.DictReader(f))
            self.assertEqual(len(grouped), 18)
            self.assertTrue(all(r['presentations'] == '12' for r in grouped))
            with (Path(folder) / 'condition_summary.csv').open() as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 6)
            self.assertTrue(all(r['families'] == '3' for r in summary))
            for row in summary:
                self.assertAlmostEqual(float(row['selection_accuracy']), 1 if row['condition'] == 'both' else 0.5)

    def test_missing_root_no_files(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / 'missing'
            argv = ['rule_evidence.py', '--experiment-dir', str(missing)]
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(SystemExit, '--experiment-dir'):
                evidence.main()
            self.assertFalse(missing.exists())

    def test_generate_smoke_resume_and_grouping(self):
        with tempfile.TemporaryDirectory() as folder:
            base = ['rule_evidence.py', '--data-dir', str(PROJECT / 'data'), '--experiment-dir', folder]
            with patch.object(sys, 'argv', base + ['--generate-only']), patch.object(evidence.experiment, 'load_native_model') as loader:
                evidence.main()
                loader.assert_not_called()
            out = Path(folder) / 'probing/qwen3_5/rule/evidence_integration'
            self.assertTrue((out / 'config.json').exists())
            self.assertFalse((out / 'predictions.csv').exists())
            adapter = SimpleNamespace(option_probs=lambda *args: {'X': 0.6, 'Y': 0.3, 'Z': 0.1})
            with patch.object(sys, 'argv', base + ['--smoke-test']), patch.dict(sys.modules, {'torch': SimpleNamespace(manual_seed=lambda seed: None)}), patch.object(evidence.experiment, 'load_native_model') as loader, patch.object(evidence.experiment, 'vlm', adapter):
                evidence.main()
                evidence.main()
                self.assertEqual(loader.call_count, 1)
            with (out / 'smoke/predictions.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            with (out / 'smoke/family_summary.csv').open() as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 9)
            self.assertTrue(all(r['presentations'] == '1' for r in summary))
            # Full design has 108 presentations, but only three test families.
            jobs = [j for f in self.families if f['split'] == 'test' for j in evidence.presentations(f)]
            self.assertEqual(len(jobs), 108)
            self.assertEqual(len({j['family_id'] for j in jobs}), 3)


if __name__ == '__main__':
    unittest.main()
