"""CPU-only protocol tests; no pretrained model weights or GPU inference."""
from collections import Counter
import copy
import contextlib
import io
import sys
import json
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_sets import make_manifest, compatible_rules
import compare_3_vs_7 as compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        (self.data / 'images').mkdir(parents=True)
        records = []
        for bits in product((0, 1), repeat=3):
            for i in range(125):
                name = f"scene_{''.join(map(str, bits))}_{i:03d}.png"
                (self.data / 'images' / name).write_bytes(b'fixture')
                records.append(dict(filename=name, bits=list(bits), split='train' if i < 100 else 'test'))
        (self.data / 'metadata.json').write_text(json.dumps(dict(records=records)))
        self.rules = Path(__file__).with_name('rules_7.json')
        self.spec = compare.validate_rules(self.rules)
        manifest = make_manifest(self.data, self.rules.with_name('rules.json'), None,
                                 {'train': 30, 'test': 10}, 10, 2026)
        self.manifest = self.root / 'probe-manifest-10.json'
        self.manifest.write_text(json.dumps(manifest))
        self.sets, _ = compare.validate(self.data, self.manifest, self.rules)

    def test_truth_tables_and_rejections(self):
        expected = ['001 011 101 110 111', '010 011 101 110 111', '011 100 101 110 111',
                    '001 010 011 100 101 111', '001 010 011 100 110 111', '001 010 100 101 110 111',
                    '001 010 011 100 101 110']
        self.assertEqual([r['true_bits'] for r in self.spec['rules']], [s.split() for s in expected])
        for kind in ('duplicate', 'invalid', 'wrong_expression'):
            spec = copy.deepcopy(self.spec)
            if kind == 'duplicate':
                spec['rules'][3]['true_bits'] = spec['rules'][0]['true_bits']
            elif kind == 'invalid':
                spec['rules'][3]['true_bits'].append('002')
            else:
                spec['rules'][3]['expression'] = 'A OR B'
            path = self.root / f'{kind}.json'
            path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                compare.validate_rules(path)

    def test_identifiability_and_exact_failure(self):
        self.assertEqual(Counter(s['rule'] for s in self.sets), {'R1': 10, 'R2': 10, 'R3': 10})
        for s in self.sets:
            for n in (3, 7):
                self.assertEqual(compatible_rules(s['bits'], self.spec['rules'][:n]), [s['rule']])
        manifest = json.loads(self.manifest.read_text())
        target = next(s for s in manifest['sets'] if s['split'] == 'test')
        target['image_files'] = [f'scene_111_{i:03d}.png' for i in range(100, 110)]
        target['bits'] = ['111'] * 10
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, f"{target['set_id']}: not uniquely identifiable among 3 rules"):
            compare.validate(self.data, self.manifest, self.rules)

    def test_pairing_rotations_and_identical_prompt_instructions(self):
        jobs = compare.jobs(self.sets, self.spec)
        self.assertEqual(len(jobs), 300)
        self.assertEqual(len(compare.jobs(self.sets, self.spec, True)), 60)
        for item in self.sets:
            for n in (3, 7):
                labels = []
                for rotation in range(n):
                    job = jobs[f"{item['set_id']}:{n}:{rotation}"]
                    self.assertEqual(job['image_files'], item['image_files'])
                    labels.append(next(k for k, v in job['option_mapping'].items() if v == item['rule']))
                self.assertEqual(set(labels), set(map(str, range(1, n+1))))
        def instructions(n):
            prompt = compare.prompt_for(self.spec, compare.mapping(n, 0))
            prefix, rest = prompt.split('Candidate rules:\n')
            rest = rest.split('\n\n', 1)[1]
            return prefix, rest.split('Reply with nothing but')[0]
        self.assertEqual(instructions(3), instructions(7))
        with self.assertRaises(ValueError):
            compare.mapping(7, 7)

    def make_rows(self, single=False):
        jobs = compare.jobs(self.sets, self.spec, single)
        rows = []
        for job in jobs.values():
            # Always position 2: one correct rotation/set and no rule consistency.
            probs = {label: .4/(job['condition']-1) for label in job['option_mapping']}
            probs['2'] = .6
            rule = job['option_mapping']['2']
            rows.append(dict(job, probabilities=probs, predicted_label='2',
                             predicted_rule=rule, correct=rule == job['true_rule'], input_tokens=100))
        return jobs, rows

    def test_set_aggregation_and_bias(self):
        _, rows = self.make_rows()
        summary = compare.summarize(rows)
        for n in (3, 7):
            condition = summary['conditions'][str(n)]
            self.assertAlmostEqual(condition['accuracy'], 1/n)
            self.assertAlmostEqual(condition['chance_adjusted_accuracy'], 0)
            self.assertEqual(condition['rotation_consistency'], 0)
            self.assertEqual(condition['answer_position_counts']['2'], 30*n)
            for row in condition['confusion_set_equivalents'].values():
                self.assertAlmostEqual(sum(row.values()), 10)
        self.assertEqual(summary['paired_comparison']['sets'], 30)
        self.assertAlmostEqual(summary['paired_comparison']['mean_accuracy_difference_7_minus_3'], 1/7-1/3)
        _, pilot = self.make_rows(single=True)
        self.assertIsNone(compare.summarize(pilot, True)['conditions']['7']['rotation_consistency'])
        with self.assertRaises(ValueError):
            compare.summarize(rows[:-1])

    def test_resume_refuses_changed_inputs_duplicates_and_partial_lines(self):
        jobs, rows = self.make_rows()
        out = self.root / 'new-results'
        config = {'protocol': 'test'}
        self.assertEqual(compare.resume_rows(out, config, jobs, False), [])
        out.mkdir()
        compare.atomic_json(out / 'config.json', dict(config, numeric_label_token_ids=list(range(7))))
        log = out / 'predictions.jsonl'
        log.write_text(json.dumps(rows[0])+'\n')
        self.assertEqual(compare.resume_rows(out, config, jobs, True), rows[:1])
        with self.assertRaises(ValueError):
            compare.resume_rows(out, config, jobs, False)
        with self.assertRaises(ValueError):
            compare.resume_rows(out, {'protocol': 'other'}, jobs, True)
        for content in (json.dumps(rows[0]), (json.dumps(rows[0])+'\n')*2):
            log.write_text(content)
            with self.assertRaises(ValueError):
                compare.resume_rows(out, config, jobs, True)
        changed = copy.deepcopy(rows[0]); changed['image_files'].reverse()
        log.write_text(json.dumps(changed)+'\n')
        with self.assertRaisesRegex(ValueError, 'image_files'):
            compare.resume_rows(out, config, jobs, True)

    def test_cli_validate_smoke_pilot_and_resume_without_gpu(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [int(text)]
        class Processor:
            tokenizer = Tokenizer()
        out = self.root / 'comparison'
        base = ['compare_3_vs_7.py', '--data-dir', str(self.data),
                '--manifest', str(self.manifest), '--out', str(out)]
        with patch.object(sys, 'argv', base+['--validate-only']), patch.object(compare.shared, 'load_model') as loader, contextlib.redirect_stdout(io.StringIO()):
            compare.main()
            loader.assert_not_called()
        self.assertFalse(out.exists())
        def fake_score(model, processor, token_ids, files, data_dir, prompt, max_pixels):
            return {str(i): 1/len(token_ids) for i in range(1,len(token_ids)+1)}, 100
        with patch.object(sys, 'argv', base+['--smoke-test']), patch.object(compare.shared, 'load_model', return_value=(None, Processor(), [])), patch.object(compare, 'evaluate_numeric', side_effect=fake_score) as scorer, contextlib.redirect_stdout(io.StringIO()):
            compare.main()
            self.assertEqual(scorer.call_count, 2)
            self.assertEqual(scorer.call_args_list[0].args[3], scorer.call_args_list[1].args[3])
        self.assertFalse(out.exists())
        with patch.object(sys, 'argv', base+['--pilot']), patch.object(compare.shared, 'load_model', return_value=(None, Processor(), [])), patch.object(compare, 'evaluate_numeric', side_effect=fake_score) as scorer, contextlib.redirect_stdout(io.StringIO()):
            compare.main()
            self.assertEqual(scorer.call_count, 60)
        snapshot = (out/'predictions.jsonl').read_bytes()
        with patch.object(sys, 'argv', base+['--pilot', '--resume']), patch.object(compare.shared, 'load_model') as loader, contextlib.redirect_stdout(io.StringIO()):
            compare.main()
            loader.assert_not_called()
        self.assertEqual(snapshot, (out/'predictions.jsonl').read_bytes())
        self.assertEqual(set(p.name for p in out.iterdir()), {'config.json','predictions.jsonl','summary.json'})

    def test_labels_reject_multitoken_and_aliases(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [int(text)+10]
        self.assertEqual(compare.label_tokens(Tokenizer()), list(range(11, 18)))
        with patch.object(Tokenizer, 'encode', return_value=[1, 2]):
            with self.assertRaises(ValueError):
                compare.label_tokens(Tokenizer())
        with patch.object(Tokenizer, 'encode', return_value=[1]):
            with self.assertRaises(ValueError):
                compare.label_tokens(Tokenizer())

    def test_cpu_numeric_scoring_and_same_preprocessing_as_existing_helper(self):
        import torch
        from PIL import Image
        files = self.sets[0]['image_files']
        for name in files:
            Image.new('RGB', (100, 200), color='green').save(self.data/'images'/name)
        class Prepared(dict):
            def to(self, device):
                return self
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [int(text)+10] if text.isdigit() else [100] + ([int(text[-1])+10] if text[-1].isdigit() else [])
        class Processor:
            tokenizer = Tokenizer()
            calls = []
            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                if kwargs.get('tokenize') is False:
                    return 'assistant:'
                return Prepared(input_ids=torch.tensor([[100, 101]]))
        class Model:
            device = 'cpu'
            def __call__(self, **kwargs):
                class Output:
                    logits = torch.arange(30, dtype=torch.float32).reshape(1, 1, 30)
                return Output()
        processor = Processor()
        ids = compare.label_tokens(processor.tokenizer)
        probabilities, tokens = compare.evaluate_numeric(Model(), processor, ids, files, self.data, 'prompt', 10000)
        expected = torch.tensor(ids, dtype=torch.float32).softmax(0).tolist()
        self.assertEqual(list(probabilities), list('1234567'))
        self.assertEqual(list(probabilities.values()), expected)
        self.assertEqual(tokens, 2)
        ours = [c['image'].size for c in processor.calls[-1][0][0]['content'][:-1]]
        compare.shared.evaluate(Model(), processor, ids[:3], files, self.data, 'prompt', 10000)
        original = [c['image'].size for c in processor.calls[-1][0][0]['content'][:-1]]
        self.assertEqual(ours, original)
        self.assertFalse(processor.calls[-1][1]['enable_thinking'])


if __name__ == '__main__':
    unittest.main()
