"""CPU audits of the published Experiment 04 artifacts and rotation analysis."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_results", ROOT / "code/04-lora/analyze_results.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
RESULTS = ROOT / "experiments/04-lora/results"


class ResultsTests(unittest.TestCase):
    def test_source_checksums_and_readme_links(self):
        import re
        provenance = json.loads((RESULTS / "provenance.json").read_text())
        for name, record in provenance["artifacts"].items():
            content = (RESULTS / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])
            self.assertEqual(len(content), record["bytes"])
        readme = RESULTS.parent / "README.md"
        for link in re.findall(r"\]\(([^)]+)\)", readme.read_text()):
            if not link.startswith("http"):
                self.assertTrue((readme.parent / link.split("#")[0]).exists(), link)

    def test_saved_artifacts_and_findings(self):
        report = analysis.analyze(RESULTS)
        self.assertEqual(report["before"]["accuracy"], 29 / 90)
        self.assertEqual(report["after"]["accuracy"], 30 / 90)
        self.assertEqual(report["changes"], dict(wrong_to_correct=11, correct_to_wrong=10, wrong_to_different_wrong=7))
        self.assertEqual(report["before"]["consistent_sets"], 6)
        self.assertEqual(report["after"]["consistent_sets"], 0)
        self.assertEqual(report["after"]["answer_labels"], {"Y": 90})
        saved = json.loads((RESULTS / "derived/analysis.json").read_text())
        self.assertEqual(report, saved)

    def test_consistency_tracks_rules_instead_of_letters(self):
        before = analysis.read_rows(RESULTS / "pretrained/predictions.jsonl")
        group = [copy.deepcopy(r) for r in before.values() if r["set_id"] == "validation_R1_00"]
        self.assertEqual(analysis.summary(group)["consistent_sets"], 1)
        self.assertEqual(len({r["answer_label"] for r in group}), 3)
        group.pop()
        with self.assertRaises(AssertionError):
            analysis.summary(group)

    def test_reject_duplicate_keys_and_incorrect_labels(self):
        row = next(iter(analysis.read_rows(RESULTS / "pretrained/predictions.jsonl").values()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.jsonl"
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                analysis.read_rows(path)
        row["answer_label"] = "X"
        with self.assertRaises(AssertionError):
            analysis.summary([row])

    def test_bootstrap_pairs_identical_models(self):
        rows = analysis.read_rows(RESULTS / "pretrained/predictions.jsonl")
        interval = analysis.paired_bootstrap(rows, rows, seed=7, draws=1000)
        self.assertEqual(interval["difference"], [0, 0])
        self.assertEqual(interval["pretrained"], interval["finetuned"])
        self.assertEqual(interval, analysis.paired_bootstrap(rows, rows, seed=7, draws=1000))


if __name__ == "__main__":
    unittest.main()
