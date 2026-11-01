"""Temporary fixtures check result transfer; these are not measured model outputs."""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'code/02-probing/copy_concept_results.py'
spec = importlib.util.spec_from_file_location('copy_results', SCRIPT)
copy_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy_results)


class CopyConceptResultsTests(unittest.TestCase):
    def test_conflicting_copy_preserves_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / 'source', Path(directory) / 'destination'
            source.write_bytes(b'original')
            destination.write_bytes(b'different')
            with self.assertRaises(ValueError):
                copy_results.copy_checked(source, destination)
            self.assertEqual(source.read_bytes(), b'original')
            self.assertEqual(destination.read_bytes(), b'different')

    def test_recovery_and_lightweight_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'checkout'
            script = project / 'code/02-probing/copy_concept_results.py'
            script.parent.mkdir(parents=True)
            script.write_bytes(SCRIPT.read_bytes())
            metadata = project / 'data/metadata.json'
            metadata.parent.mkdir()
            metadata.write_bytes((PROJECT / 'data/metadata.json').read_bytes())
            test = [r for r in json.loads(metadata.read_text())['records'] if r['variation'] >= 4]
            for model, model_id in [('qwen3', 'Qwen/Qwen3-VL-4B-Instruct'), ('qwen3_5', 'Qwen/Qwen3.5-4B')]:
                folder = root / 'old/probing' / model / 'concept'
                folder.mkdir(parents=True)
                config = dict(model=model_id, task='concept', smoke_test=False,
                              metadata_hash=hashlib.sha256(metadata.read_bytes()).hexdigest()[:16])
                (folder / 'config.json').write_text(json.dumps(config))
                Image.new('RGB', (4, 4)).save(folder / 'concept_accuracy.png')
                with (folder / 'concept_predictions.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['layer', 'pool', 'control', 'filename', 'truth', 'predicted', 'correct', 'true_class_probability'])
                    writer.writeheader()
                    for pool in ('visual_mean', 'text_mean', 'last'):
                        for control in ('measured', 'shuffled_train_labels'):
                            for record in test:
                                label = ''.join(map(str, record['bits']))
                                writer.writerow(dict(layer=1, pool=pool, control=control, filename=record['filename'], truth=label, predicted=label, correct=1, true_class_probability=0.5))
                (folder / 'scene_000_00.npz').write_bytes(b'checkpoint fixture')
            command = [sys.executable, str(script), '--drive-results', str(root / 'drive'), '--recover-from', str(root / 'old')]
            for _ in range(2):
                subprocess.run(command, check=True, capture_output=True, text=True)
            for model in ('qwen3', 'qwen3_5'):
                original = root / 'old/probing' / model / 'concept'
                drive = root / 'drive/probing' / model / 'concept'
                github = project / 'results/02-probing' / model / 'concept'
                self.assertTrue((original / 'scene_000_00.npz').exists())
                self.assertEqual((original / 'scene_000_00.npz').read_bytes(), (drive / 'scene_000_00.npz').read_bytes())
                self.assertEqual({p.name for p in github.iterdir()}, set(copy_results.FILES))
                for name in copy_results.FILES:
                    self.assertEqual((original / name).read_bytes(), (github / name).read_bytes())
                with (github / 'concept_predictions.csv').open(newline='') as f:
                    rows = list(csv.DictReader(f))
                rows[0]['correct'] = '0'
                with (github / 'concept_predictions.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    copy_results.verify_results(github, model, metadata)


if __name__ == '__main__':
    unittest.main()
