"""Tests use made-up rows, never measured research results."""

import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "code/03-activation-patching")
)
from analyze_baseline import probabilities, validate_rows


class AnalysisTests(unittest.TestCase):
    def row(self):
        return dict(
            donor_rule="R2",
            receiver_rule="R1",
            receiver_prediction="R1",
            receiver_p_donor_rule="0.3",
            receiver_p_receiver_rule="0.5",
        )

    def test_reconstruct_distinct_targets(self):
        self.assertAlmostEqual(probabilities(self.row(), "receiver")["R3"], 0.2)

    def test_same_rule_missing_values_stay_unknown(self):
        row = self.row()
        row.update(donor_rule="R1", receiver_p_donor_rule=".5")
        self.assertEqual(probabilities(row, "receiver"), dict(R1=0.5, R2=None, R3=None))

    def test_inconsistent_scores_rejected(self):
        row = self.row()
        row["receiver_p_donor_rule"] = ".8"
        with self.assertRaises(ValueError):
            probabilities(row, "receiver")
        row = self.row()
        row["receiver_prediction"] = "R2"
        with self.assertRaises(ValueError):
            probabilities(row, "receiver")

    def test_incomplete_run_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([], [])


class SavedRunTests(unittest.TestCase):
    def test_actual_full_run_and_corrupt_record(self):
        import csv
        import json
        root = Path(__file__).resolve().parents[1] / 'experiments/03-activation-patching/baseline-analysis'
        with (root/'raw/patches.csv').open() as f:
            rows = list(csv.DictReader(f))
        config = json.loads((root/'raw/config.json').read_text())
        validate_rows(rows, config['pairs'])
        self.assertEqual(len(rows), 3456)
        with self.assertRaises(ValueError):
            validate_rows(rows + [rows[0]], config['pairs'])
        final = [r for r in rows if r['layer']=='32' and r['position']=='final']
        self.assertEqual(sum(r['patched_prediction']==r['donor_prediction'] for r in final),54)
        different = [r for r in final if r['condition']=='different']
        self.assertEqual(sum(int(r['changed_answer']) for r in different),16)
        self.assertEqual(sum(int(r['changed_to_donor_rule']) for r in different),7)
        manifest=json.loads((root/'derived/input_manifest.json').read_text())
        self.assertEqual(len(manifest),27)
        for item in manifest.values():
            self.assertEqual(len(item['images']),10)
            self.assertEqual(len(set(item['images'])),10)

    def test_raw_checksums_and_script_identity(self):
        import json
        from analyze_baseline import checksum
        root = Path(__file__).resolve().parents[1] / 'experiments/03-activation-patching/baseline-analysis'
        provenance=json.loads((root/'derived/provenance.json').read_text())
        for name,value in provenance['source_checksums'].items():
            self.assertEqual(checksum(root/'raw'/name),value)
        # The provenance describes the original analysis, before the notebook change.
        script=root/'derived/analyze_baseline_original.py.txt'
        self.assertEqual(checksum(script),provenance['analysis_script_checksum'])


if __name__ == "__main__":
    unittest.main()
