"""Logical pairing and CPU toy-model checks; not Qwen GPU validation."""
import csv
import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'code/03-activation-patching'))
import patch_rules as patcher
try:
    import torch
except ImportError:
    torch = None


class PairingTests(unittest.TestCase):
    def test_missing_results_root_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing'
            result = subprocess.run([sys.executable, patcher.__file__, '--experiment-dir', str(missing)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Mount Google Drive', result.stderr)
            self.assertFalse(missing.exists())

    def test_pairs_are_matched_and_identifiable(self):
        records = json.loads((PROJECT / 'data/metadata.json').read_text())['records']
        pairs = patcher.make_pairs(records)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs, patcher.make_pairs(records))
        for pair in pairs:
            left, right = pair['rules']
            a, b = pair['scenes'][left], pair['scenes'][right]
            self.assertEqual(sum(x == y for x, y in zip(a['image_files'], b['image_files'])), 9)
            for name, scene in pair['scenes'].items():
                items = [{'bits': bits} for bits in scene['scene_bits']]
                self.assertEqual(patcher.experiment.compatible_rules(items), [name[:2]])
            for mapping, display in patcher.experiment.layouts('cyclic'):
                patcher.experiment.IMAGE_DIR = PROJECT / 'data/images'
                self.assertEqual(patcher.experiment.build_input(a, mapping, display)[1],
                                 patcher.experiment.build_input(b, mapping, display)[1])
        with self.assertRaises(ValueError):
            patcher.make_pairs(records, 7)


@unittest.skipIf(torch is None, 'CPU PyTorch required for hook tests')
class HookTests(unittest.TestCase):
    def setUp(self):
        class Block(torch.nn.Module):
            def __init__(self, tuple_output):
                super().__init__()
                self.tuple_output = tuple_output
            def forward(self, hidden):
                output = hidden + hidden.mean(dim=1, keepdim=True) * 0.1
                return (output, 'unchanged') if self.tuple_output else output

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(False), Block(True), Block(False)])
                self.model = SimpleNamespace(layers=self.blocks)
            def forward(self, hidden, use_cache=False, logits_to_keep=1):
                for block in self.blocks:
                    hidden = block(hidden)
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                return SimpleNamespace(logits=hidden[:, -1:, :])
        model = Model()
        def inputs(paths, prompt):
            n = len(paths)
            return {'hidden': torch.arange(n * 4, dtype=torch.float32).reshape(1, n, 4) / 10}
        self.adapter = SimpleNamespace(model=model, _inputs=inputs)
        self.ids = dict(X=0, Y=1, Z=2)

    def test_self_and_final_transfer_with_different_lengths(self):
        donor = patcher.forward(self.adapter, ['a'] * 5, '', self.ids, capture=True)
        recipient = patcher.forward(self.adapter, ['b'] * 3, '', self.ids, capture=True)
        plain = patcher.forward(self.adapter, ['b'] * 3, '', self.ids)
        torch.testing.assert_close(plain['logits'], recipient['logits'])
        for layer in range(3):
            saved = recipient['states'][layer].clone()
            patched = patcher.forward(self.adapter, ['b'] * 3, '', self.ids, layer, saved)
            torch.testing.assert_close(patched['logits'], recipient['logits'])
            torch.testing.assert_close(saved, recipient['states'][layer])
        patched = patcher.forward(self.adapter, ['b'] * 3, '', self.ids, 2, donor['states'][2])
        torch.testing.assert_close(patched['logits'], donor['logits'])
        self.assertTrue(all(not block._forward_hooks for block in self.adapter.model.blocks))

    def test_cleanup_on_invalid_state_and_dtype_conversion(self):
        with self.assertRaises(ValueError):
            patcher.forward(self.adapter, ['a'], '', self.ids, 1, torch.zeros(9))
        self.assertTrue(all(not block._forward_hooks for block in self.adapter.model.blocks))
        state = torch.ones(4, dtype=torch.float64)
        saved = state.clone()
        result = patcher.forward(self.adapter, ['a'], '', self.ids, 1, state)
        self.assertEqual(result['logits'].dtype, torch.float32)
        torch.testing.assert_close(state, saved)

    def test_smoke_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['patch_rules.py', '--data-dir', str(PROJECT / 'data'), '--experiment-dir', directory, '--smoke-test']
            with patch.object(sys, 'argv', argv), patch.object(patcher.experiment, 'load_native_model'), patch.object(patcher.experiment, 'vlm', self.adapter), patch.object(patcher.experiment, 'TOKEN_IDS', self.ids), patch.object(patcher, 'save_plot'), patch.object(patcher, 'forward', wraps=patcher.forward) as run:
                patcher.main()
                self.assertEqual(run.call_count, 22)  # Four baselines and 18 patches.
                patcher.main()
                self.assertEqual(run.call_count, 22)
            out = Path(directory) / 'activation-patching/qwen3_5/smoke'
            with (out / 'patches.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 18)
            self.assertEqual({r['control'] for r in rows}, {'self', 'cross', 'same_rule'})
            self.assertEqual(len(list(out.glob('*.pt'))), 4)
            self.assertEqual(json.loads((out / 'numerical_checks.json').read_text())['failed_rows'], 0)
            with (out / 'baselines.csv').open() as f:
                baselines = list(csv.DictReader(f))
            self.assertEqual(len(baselines), 4)
            for row in baselines:
                self.assertAlmostEqual(sum(float(row[f'probability_{label}']) for label in 'XYZ'), 1, places=6)

    def test_full_scan_covers_pairs_directions_layouts_and_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['patch_rules.py', '--data-dir', str(PROJECT / 'data'), '--experiment-dir', directory]
            with patch.object(sys, 'argv', argv), patch.object(patcher.experiment, 'load_native_model'), patch.object(patcher.experiment, 'vlm', self.adapter), patch.object(patcher.experiment, 'TOKEN_IDS', self.ids), patch.object(patcher, 'save_plot'):
                patcher.main()
            out = Path(directory) / 'activation-patching/qwen3_5'
            with (out / 'patches.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 162)
            self.assertEqual(len({r['pair_id'] for r in rows}), 3)
            self.assertEqual({r['layout'] for r in rows}, {'0', '1', '2'})
            self.assertEqual({r['layer'] for r in rows}, {'1', '2', '3'})
            for pair in {r['pair_id'] for r in rows}:
                directions = {(r['recipient_rule'], r['donor_rule']) for r in rows if r['pair_id'] == pair}
                self.assertEqual(len(directions), 2)
            checks = json.loads((out / 'numerical_checks.json').read_text())
            self.assertEqual(checks['self_checks'], 54)
            self.assertEqual(checks['final_checks'], 18)
            self.assertEqual(checks['failed_rows'], 0)

    def test_failed_control_stops_and_is_not_skipped_on_resume(self):
        original = patcher.effect_row
        def failed(*args, **kwargs):
            row = original(*args, **kwargs)
            if row['control'] == 'self':
                row['self_check_pass'] = False
            return row
        with tempfile.TemporaryDirectory() as directory:
            argv = ['patch_rules.py', '--data-dir', str(PROJECT / 'data'), '--experiment-dir', directory, '--smoke-test']
            with patch.object(sys, 'argv', argv), patch.object(patcher.experiment, 'load_native_model') as loader, patch.object(patcher.experiment, 'vlm', self.adapter), patch.object(patcher.experiment, 'TOKEN_IDS', self.ids), patch.object(patcher, 'effect_row', side_effect=failed):
                with self.assertRaisesRegex(SystemExit, 'scan stopped'):
                    patcher.main()
                with self.assertRaisesRegex(SystemExit, 'saved numerical control failed'):
                    patcher.main()
                self.assertEqual(loader.call_count, 1)
            out = Path(directory) / 'activation-patching/qwen3_5/smoke'
            self.assertEqual(json.loads((out / 'numerical_checks.json').read_text())['failed_rows'], 1)

    def test_logit_difference_and_flags(self):
        pair = dict(pair_id='pair', last_layer=2)
        mapping = dict(X='R1', Y='R2', Z='R3')
        row = patcher.effect_row(pair, 0, mapping, 'R1', 'R2', 'cross', 2,
                                 np.array([3, 1, 0]), np.array([1, 4, 0]), np.array([1, 4, 0]), 0.05)
        self.assertAlmostEqual(row['margin_change'], 5)
        self.assertTrue(row['final_transfer_pass'])
        self.assertFalse(row['same_baseline_prediction'])


if __name__ == '__main__':
    unittest.main()
