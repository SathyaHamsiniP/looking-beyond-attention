"""CPU checks for the pilot split and resumable extraction orchestration."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'code/02-probing'))
sys.path.insert(0, str(PROJECT / 'code/01-operator-controls'))
import probe_rules as probe


class RuleProbeTests(unittest.TestCase):
    def setUp(self):
        self.experiment = importlib.import_module('operator_controls_qwen35')
        self.experiment.configure_data(PROJECT / 'data', Path('/tmp/unused-rule-output'))

    def test_split_is_deterministic_image_disjoint_and_identifiable(self):
        original = self.experiment.POOL
        sets = probe.make_sets(self.experiment)
        self.assertIs(self.experiment.POOL, original)
        self.assertEqual(sets, probe.make_sets(self.experiment))
        report = probe.split_report(sets)
        self.assertEqual(report['train']['unique_underlying_images'], 28)
        self.assertEqual(report['test']['unique_underlying_images'], 14)
        self.assertEqual(report['test']['unique_sets'], 3)
        self.assertEqual(report['test']['max_mutually_image_disjoint_sets'], 1)
        train = {n for s in sets if s['split'] == 'train' for n in s['image_files']}
        test = {n for s in sets if s['split'] == 'test' for n in s['image_files']}
        self.assertFalse(train & test)
        self.assertEqual(len({tuple(sorted(s['image_files'])) for s in sets}), 27)
        for scene in sets:
            self.assertEqual(probe.diagnostic_rule(scene), scene['rule'])
            self.assertAlmostEqual(sum(probe.scene_counts(scene)), 1)
            self.assertEqual(len(scene['image_files']), 10)
        self.assertNotIn('R1', probe.PROMPT)
        self.assertNotIn('(A AND B)', probe.PROMPT)

    def test_smoke_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            command = ['probe_rules.py', '--model', 'qwen3_5', '--data-dir', str(PROJECT / 'data'),
                       '--experiment-dir', folder, '--smoke-test']
            # Fake arrays test orchestration only; never report them as model results.
            with patch.object(sys, 'argv', command), patch.dict(sys.modules, {'torch': SimpleNamespace(manual_seed=lambda seed: None)}), patch.object(self.experiment, 'load_native_model'), patch.object(probe, 'capture', return_value=np.zeros((2, 3, 4))) as capture, patch.object(probe, 'score_rules') as score:
                probe.main()
                probe.main()
                self.assertEqual(capture.call_count, 2)
                score.assert_not_called()
            out = Path(folder) / 'probing/qwen3_5/rule/smoke'
            self.assertEqual(len(list(out.glob('*.npz'))), 2)
            self.assertFalse((out.parent / 'config.json').exists())
            config = json.loads((out / 'config.json').read_text())
            self.assertEqual(len(config['sets']), 2)
            with patch.object(sys, 'argv', command + ['--precision', 'fp32']):
                with self.assertRaisesRegex(ValueError, 'Checkpoint settings differ'):
                    probe.main()


if __name__ == '__main__':
    unittest.main()
