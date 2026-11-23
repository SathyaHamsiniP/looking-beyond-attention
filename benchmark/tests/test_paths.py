"""CPU checks for moved entry points and unchanged source/checkpoint guards."""
import setup_paths
from benchmark_paths import CODE, REPO, source_path
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lora_7rules as seven
import seven_rule_validation as validation
import test_lora_7rules as fixtures


class PathTests(unittest.TestCase):
    def test_every_moved_module_imports_and_entrypoint_has_help(self):
        scripts=list(CODE.rglob('*.py'))
        for path in scripts:
            if '__pycache__' in path.parts:continue
            module=importlib.import_module(path.stem)
            self.assertEqual(Path(module.__file__).resolve(),path.resolve())
            if 'def main(' in path.read_text():
                result=subprocess.run([sys.executable,str(path),'--help'],cwd=REPO,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,(path,result.stderr))
        with self.assertRaises(ValueError):source_path('does-not-exist.py')

    def test_all_four_seven_rule_cpu_commands(self):
        fixtures.SevenRuleTests.setUpClass()
        try:
            fixture=fixtures.SevenRuleTests;data=fixture.data;m=fixture.manifest
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);run=root/'lora';dataset=run/'dataset';dataset.mkdir(parents=True)
                seven.atomic_json(dataset/'manifest.json',m)
                (dataset/'partitions.json').write_bytes(fixture.mapping_bytes)
                baseline=root/'baseline';baseline.mkdir()
                seven.atomic_json(baseline/'config.json',seven.configuration(m,'pretrained'))
                jobs=seven.evaluation_jobs(m,'validation');rows=[]
                for job in jobs.values():
                    label='7';pred=job['option_mapping'][label]
                    probs={l:.4/6 for l in seven.LABELS};probs[label]=.6
                    rows.append(dict(job,probabilities=probs,predicted_label=label,predicted_rule=pred,correct=pred==job['true_rule']))
                (baseline/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                manifest=dataset/'manifest.json'
                commands=[
                    [str(source_path('lora_7rules.py')),'train','--data-dir',str(data),'--out',str(run),'--validate-only'],
                    [str(source_path('operator_controls_7rules.py')),'--manifest',str(manifest),'--data-dir',str(data),'--baseline',str(baseline),'--out',str(root/'controls'),'--validate-only'],
                    [str(source_path('activation_patching_7rules.py')),'--manifest',str(manifest),'--data-dir',str(data),'--cache',str(baseline),'--out',str(root/'patch')],
                    [str(source_path('probing_7rules.py')),'--manifest',str(manifest),'--data-dir',str(data),'--out',str(root/'probe'),'--validate-only']]
                for args in commands:
                    result=subprocess.run([sys.executable,*args],cwd=REPO,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,(args,result.stderr))
                # Keep the old source records distinct; never normalize them to
                # current hashes merely to reuse predictions under moved code.
                cfg=seven.read(baseline/'config.json')
                cfg['source_hashes']['compare_3_vs_7.py']='0'*64
                seven.atomic_json(baseline/'config.json',cfg)
                view=validation.validation_view(manifest,data)
                with self.assertRaisesRegex(ValueError,'Inference source changed'):
                    validation.verify_baseline(baseline,view)
        finally:fixtures.SevenRuleTests.tearDownClass()

    def test_original_rules_results_and_historical_hook_paths(self):
        import activation_patching_7rules as patching
        import probing_7rules as probing
        self.assertTrue(patching.HISTORICAL.is_file())
        self.assertEqual(patching.HISTORICAL,REPO/'three-operand-experiments/code/03-activation-patching/patch_activations_qwen35.py')
        self.assertEqual(Path(probing.capture_helpers().__file__).resolve(),source_path('probing.py'))
        self.assertEqual(seven.RULES_PATH,source_path('rules_7.json'))
        self.assertIn('benchmark_paths.py',seven.code_identity())
        with tempfile.TemporaryDirectory() as tmp:
            seven.atomic_json(Path(tmp)/'config.json',dict(source_hashes={'probing_7rules.py':'old'}))
            with self.assertRaisesRegex(ValueError,'Extraction source changed'):
                probing.fit(Path(tmp),[],{},0)
        with self.assertRaisesRegex(ValueError,'outside Git'):
            validation.local_path(REPO/'benchmark/results/new-output')


if __name__=='__main__':unittest.main()
