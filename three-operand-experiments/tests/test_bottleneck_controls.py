"""GPU-free validation for Experiment 04."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/04-bottleneck-controls/run_bottleneck_controls.py"
spec = importlib.util.spec_from_file_location("bottleneck_controls", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BottleneckControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = json.loads((ROOT / "data/metadata.json").read_text())["records"]
        cls.families = mod.evidence.make_dataset(records)

    def test_job_design(self):
        quick = mod.build_jobs(self.families)
        smoke = mod.build_jobs(self.families, smoke=True)
        self.assertEqual(len(quick), 87)
        self.assertEqual(len(smoke), 11)
        mod.validate_design(self.families, quick)
        mod.validate_design(self.families, smoke)

    def test_prompts(self):
        job = mod.build_jobs(self.families)[0]["job"]
        for mode in mod.MODES:
            p = mod.make_prompt(job, mode, "choice")
            self.assertIn("Positive means", p)
            self.assertEqual(sum(f"{x}:" in p for x in "XYZ"), 3)
        binding = mod.make_prompt(job, "interleaved", "binding")
        self.assertNotIn("1: 010 negative", binding)
        self.assertIn("3-bit ABC", binding)

    def test_parsers(self):
        self.assertEqual(mod.parse_yes_no("YES"), "YES")
        self.assertEqual(mod.parse_yes_no("No, because"), "NO")
        self.assertIsNone(mod.parse_yes_no("Maybe"))
        self.assertEqual(mod.parse_rule("R2", "rule_id"), "R2")
        self.assertEqual(mod.parse_rule("(A AND C) OR B", "expression"), "R2")
        labels = mod.parse_binding(
            "1: 010 negative\n2: 100 negative\n3: 111 positive\n4: 000 negative"
        )
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[2], ("100", "negative"))

    def test_scoring(self):
        job = next(j for j in mod.build_jobs(self.families)
                   if j["task"] == "choice")["job"]
        row = mod.result_for({"id": "a", "job": job, "task": "choice",
                              "mode": "oracle"},
                             {"R1": 0.9, "R2": 0.05, "R3": 0.05})
        self.assertAlmostEqual(row["correct"], 1.0)
        self.assertEqual(row["chosen_label"], "X")
        chk = mod.result_for({"id": "b", "job": job, "task": "verify",
                              "mode": "oracle", "candidate": "R2", "readout": "normal"}, {"YES": 0.1, "NO": 0.9})
        self.assertEqual(chk["correct"], 1)


if __name__ == "__main__":
    unittest.main()
