"""CPU checks. Model responses in these tests are fixtures."""

import setup_paths
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evidence_design import BITS, RULES, design, solve
from evidence_metrics import baselines, paired_delta, summarize
from evidence_run import checkpoint, jobs, parse, prompt

POOL = {b: [b + "a.png", b + "b.png"] for b in BITS}


class DesignTests(unittest.TestCase):
    def test_solver_and_monotonicity_exhaustive(self):
        for n in range(9):
            for bits in combinations(BITS, n):
                expected = {
                    r["id"] for r in RULES if all(b in r["true_bits"] for b in bits)
                }
                self.assertEqual(set(solve(bits)), expected)
                for b in BITS:
                    self.assertLessEqual(set(solve((*bits, b))), expected)
        path = ["111", "011", "001", "110", "101"]
        self.assertEqual(
            [solve(path[:i]) for i in range(1, 6)],
            [[f"R{i}" for i in range(1, n + 1)] for n in (6, 5)]
            + [["R1", "R4", "R5"], ["R1", "R5"], ["R1"]],
        )

    def test_generated_designs(self):
        for number in (11, 12):
            rows = design(number, POOL, 202611, 1)
            self.assertEqual(rows, design(number, POOL, 202611, 1))
            self.assertNotEqual(rows, design(number, POOL, 11, 1))
            groups = defaultdict(list)
            for r in rows:
                self.assertEqual(r["answers"], solve(r["bits"]))
                self.assertEqual(len(set(r["bits"])), len(r["bits"]))
                self.assertTrue(
                    all(f in POOL[b] for f, b in zip(r["image_files"], r["bits"]))
                )
                groups[r["group"]].append(r)
            if number == 11:
                self.assertEqual({len(r["answers"]) for r in rows}, set(range(1, 7)))
                self.assertEqual(
                    sorted(v[-1]["answers"][0] for v in groups.values()),
                    [r["id"] for r in RULES],
                )
                for chain in groups.values():
                    for a, b in zip(chain, chain[1:]):
                        self.assertEqual(a["image_files"], b["image_files"][:-1])
                continue
            self.assertEqual(
                Counter(v[0]["condition"] for v in groups.values()),
                dict(context=18, replacement=21, control=7),
            )
            for a, b in groups.values():
                kind = a["condition"]
                self.assertEqual(len(a["answers"]), 1)
                self.assertEqual(len(a["bits"]), len(b["bits"]))
                changes = sum(
                    f != g for f, g in zip(a["image_files"], b["image_files"])
                )
                self.assertEqual(a["answers"] == b["answers"], kind == "control")
                if kind == "context":
                    i = a["focal"]
                    self.assertEqual(a["image_files"][i], b["image_files"][i])
                    self.assertGreaterEqual(changes, 2)
                    self.assertEqual(
                        baselines(a)["single_image"], baselines(b)["single_image"]
                    )
                else:
                    self.assertEqual(changes, 1)
                    self.assertEqual(
                        sum(x != y for x, y in zip(a["bits"], b["bits"])),
                        int(kind == "replacement"),
                    )

    def test_parsing_and_prompt(self):
        self.assertEqual(parse(" 1, 7 ", 1, 11), ["R1", "R2"])
        self.assertEqual(parse("7", 1, 12), ["R1"])
        for response in ("1,1", "1 and 2", "8", "1,", "R3", "", "2."):
            self.assertIsNone(parse(response, 0, 11))
        self.assertIsNone(parse("1,2", 0, 12))
        row = design(12, POOL, 12, 1)[0]
        visual = prompt(row, 0, "visual", 12)
        self.assertTrue(all(b not in visual for b in BITS))
        self.assertNotIn("A=", visual)
        self.assertIn("A=", prompt(row, 0, "symbolic", 12))
        self.assertNotEqual(visual, prompt(row, 1, "visual", 12))

    def test_checkpoint_and_metrics(self):
        rows = design(12, POOL, 42, 1)[:2]
        tasks = list(jobs({"sets": rows}, "visual", 7))
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            calls = []

            def infer(row):
                calls.append(row)
                return str((int(row["answers"][0][1:]) - 1 - row["rotation"]) % 7 + 1)

            checkpoint(folder, tasks, 12, infer, limit=1)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                checkpoint(folder, tasks, 12, None)
            full = checkpoint(folder, tasks, 12, infer)
            self.assertEqual(len(calls), 14)
            self.assertEqual(checkpoint(folder, tasks, 12, None), full)
            scores = summarize(full)
            self.assertEqual(scores["sets"], 2)
            self.assertEqual(scores["rates"]["order_agreement"], 1)
            self.assertEqual(scores["rates"]["incorrect_switch/control"], 0)
            self.assertEqual(paired_delta(full, full)["mean"], 0)
            full[0]["prediction"] = None
            self.assertAlmostEqual(summarize(full)["rates"]["exact"], 13 / 14)

    def test_elimination_and_reintroduction(self):
        rows = design(11, POOL, 42, 1)
        chain = [
            dict(r, rotation=0, prediction=r["answers"].copy())
            for r in rows
            if r["group"] == rows[0]["group"]
        ]
        self.assertEqual(summarize(chain)["rates"]["correct_elimination"], 1)
        invalid = next(r["id"] for r in RULES if r["id"] not in chain[-2]["answers"])
        chain[-1]["prediction"].append(invalid)
        self.assertGreater(summarize(chain)["rates"]["invalid_rule_reintroduced"], 0)

    def test_export_requires_all_matched_conditions(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import evidence_run as runner

        for number in (11, 12):
            with self.subTest(number=number), TemporaryDirectory() as tmp:
                out = Path(tmp)
                manifest = dict(sets=design(number, POOL, 42, 1), hashes={})
                args = SimpleNamespace(out=out, data=out, rotations=1, limit=1)
                adapter = runner.seven.read(
                    runner.RESULTS / "lora/training/complete.json"
                )["adapter_identity"]
                with (
                    patch.object(runner, "version", return_value="test"),
                    patch.object(
                        runner.seven, "adapter_identity", return_value=adapter
                    ),
                    patch.object(runner, "load_chat") as load,
                ):
                    load.return_value.chat.return_value = "1"
                    for args.model in runner.MODELS:
                        args.adapter = None if args.model == "pretrained" else out
                        for args.mode in runner.MODES:
                            args.limit = 1
                            runner.run(number, args, manifest)
                            with self.assertRaisesRegex(ValueError, "Incomplete"):
                                runner.report(number, out, manifest)
                            args.limit = None
                            runner.run(number, args, manifest)
                            load.reset_mock()
                            runner.run(number, args, manifest)
                            load.assert_not_called()
                    runner.report(number, out, manifest)
                    summary = runner.seven.read(out / "comparison.json")
                    self.assertEqual(
                        summary["paired_differences"]["lora-minus-pretrained/visual"][
                            "mean"
                        ],
                        0,
                    )
                    self.assertGreater((out / "comparison.png").stat().st_size, 0)
                    args.rotations = 7
                    with self.assertRaisesRegex(ValueError, "Incompatible"):
                        runner.run(number, args, manifest)
