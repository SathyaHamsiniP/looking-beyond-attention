"""CPU-only regression checks for the notebook-to-script conversion.

Run from project root:
    python -m unittest discover -s tests -v
These tests mock Qwen outputs; they are NOT GPU-inference verification.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'code' / 'exp04_qwen35.py'
spec = importlib.util.spec_from_file_location('exp04_qwen35', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DummyVLM:
    def chat(self, images, prompt, max_new_tokens=150, temperature=0.0):
        if 'Reply exactly on two lines: D=' in prompt:
            return 'D=0\nY=0'
        return 'A=0, B=0, C=0'


def fake_option_probs(vlm, paths, prompt, token_ids):
    if 'The correct answer is Y' in prompt:
        return {'X': 0.1, 'Y': 0.8, 'Z': 0.1}
    if 'The correct answer is Z' in prompt:
        return {'X': 0.1, 'Y': 0.1, 'Z': 0.8}
    return {'X': 0.7, 'Y': 0.2, 'Z': 0.1}


class Exp04Qwen35ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        data = base / 'data'
        images = data / 'images'
        images.mkdir(parents=True)
        records = []
        for a, b, c in mod.product((0, 1), repeat=3):
            for variation in range(6):
                filename = f'scene_{a}{b}{c}_{variation:02d}.png'
                (images / filename).touch()
                records.append({'filename': filename, 'bits': [a, b, c], 'variation': variation})
        (data / 'metadata.json').write_text(json.dumps({'records': records}), encoding='utf8')
        mod.configure_data(data, base / 'runs')
        mod.vlm = DummyVLM()
        mod.TOKEN_IDS = {'X': 100, 'Y': 200, 'Z': 300}
        mod.option_probs = fake_option_probs
        mod.MODEL_ID = 'Qwen/Qwen3.5-4B'

    def test_sample_coverage_and_seed_determinism(self):
        for rule in mod.RULES:
            for idx in range(8):
                first = mod.sample_set(rule, idx)
                second = mod.sample_set(rule, idx)
                self.assertEqual(first['image_files'], second['image_files'])
                self.assertTrue(first['identifiable'])
                self.assertEqual(first['consistent_candidates'], [rule])
                self.assertEqual(len(first['image_files']), 10)
                self.assertEqual(first['distinct_types_shown'], 5)
        self.assertEqual(len(mod.layouts('cyclic')), 3)
        self.assertEqual(len(mod.layouts('position')), 6)
        self.assertEqual(len(mod.layouts('relabel')), 6)

    def test_score_trial_and_prompt_same_options(self):
        scene = mod.sample_set('R1', 0)
        mapping, display = mod.layouts('fixed')[0]
        _, prompt = mod.build_input(scene, mapping, display)
        self.assertIn('A is true when a green cube is present.', prompt)
        self.assertIn('X: (A AND B) OR C', prompt)
        self.assertIn('Y: (A AND C) OR B', prompt)
        self.assertIn('Z: (B AND C) OR A', prompt)
        trial = mod.score_trial(scene, mapping, display)
        self.assertEqual(trial['chosen_rule'], 'R1')
        self.assertEqual(trial['chosen_label'], 'X')
        self.assertTrue(trial['correct'])
        self.assertAlmostEqual(sum(trial['probabilities'].values()), 1.0)

    def test_main_72_and_resume_skip_completed_trials(self):
        calls = []
        original = mod.option_probs

        def tracked(*args):
            calls.append(1)
            return original(*args)

        mod.option_probs = tracked
        rows, summary, path = mod.run_experiment('main_72')
        self.assertEqual(len(rows), 72)
        self.assertEqual(len(calls), 72)
        self.assertEqual(summary['trials'], 72)
        self.assertEqual(summary['independent_scene_sets'], 24)
        self.assertTrue((path / 'config.json').is_file())
        self.assertTrue((path / 'runs.jsonl').is_file())
        self.assertTrue((path / 'summary.json').is_file())
        rows2, summary2, _ = mod.run_experiment('main_72')
        self.assertEqual(len(rows2), 72)
        self.assertEqual(summary2['trials'], 72)
        self.assertEqual(len(calls), 72, 'Rerun should not repeat any Qwen inferences')

    def test_each_experiment_runs_with_fake_model(self):
        for name in mod.EXPERIMENTS:
            if name == 'main_72':
                continue
            with self.subTest(experiment=name):
                mod.run_selected(name)
                folder = mod.OUT / name
                self.assertTrue((folder / 'runs.jsonl').is_file())
                self.assertTrue((folder / 'summary.json').is_file())
                self.assertTrue((folder / 'config.json').is_file())
                self.assertGreater(len((folder / 'runs.jsonl').read_text().splitlines()), 0)

    def test_label_sanity_helper(self):
        mod.label_sanity()  # should run without raising


if __name__ == '__main__':
    unittest.main()
