"""Output checks must fail before creating a missing Drive results root."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root


class ResultsPathsTests(unittest.TestCase):
    def test_existing_root_and_missing_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(require_results_root(root), root.resolve())
            missing = root / 'not-mounted/results'
            with self.assertRaisesRegex(SystemExit, '--experiment-dir'):
                require_results_root(missing)
            self.assertFalse(missing.parent.exists())
            file = root / 'file'
            file.touch()
            with self.assertRaises(SystemExit):
                require_results_root(file)

    def test_all_runners_reject_missing_root_before_loading_models(self):
        scripts = [
            ('01-operator-controls/operator_controls_qwen3.py', []),
            ('01-operator-controls/operator_controls_qwen35.py', []),
            ('02-probing/probe_concepts.py', ['--model', 'qwen3', '--data-dir', str(CODE.parent / 'data')]),
            ('02-probing/probe_concepts.py', ['--model', 'qwen3_5', '--data-dir', str(CODE.parent / 'data')]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / 'missing/results'
            for script, extra in scripts:
                with self.subTest(script=script, extra=extra):
                    result = subprocess.run([sys.executable, str(CODE / script), *extra,
                                             '--experiment-dir', str(missing)], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Mount Google Drive', result.stderr)
                    self.assertIn('--experiment-dir', result.stderr)
                    self.assertFalse(missing.parent.exists())

    def test_default_root(self):
        self.assertEqual(str(DEFAULT_RESULTS_ROOT), '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results')


if __name__ == '__main__':
    unittest.main()
