"""Run all notebook cells on saved results without model libraries."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class NotebookSmokeTests(unittest.TestCase):
    def test_all_cells_on_cpu(self):
        root = Path(__file__).resolve().parents[1] / 'results/03-activation_patching'
        notebook = json.loads((root / 'explore_baseline.ipynb').read_text())
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        source += '\nimport sys\nassert "torch" not in sys.modules\nassert "transformers" not in sys.modules\n'
        result = subprocess.run([sys.executable, '-c', source], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Loaded 3,456 saved trials', result.stdout)
        self.assertIn('reconstructed', result.stdout)
        self.assertIn('unavailable', result.stdout)
