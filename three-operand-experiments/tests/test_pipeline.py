"""CPU scientific-invariant and real PEFT integration tests; no pretrained weights."""

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from itertools import product
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(PROJECT / "code/04-lora"))
from shared.data_loader import prompt, rotated, shortcut_control
from shared.evaluation import metrics
from shared.model_qwen35 import answer_log_likelihood, score
from shared.patching import capture, make_pairs, patched
from shared.results_paths import LEGACY_PATCHING
from shared.run_utils import Journal, atomic_json, digest, prepare_run
from tasks.build_sets import build
from tasks.rules import consistent, library, truth_tables
from tasks.validate_sets import validate
from train_lora import check_trainable, select_targets


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name)
        (cls.data / "images").mkdir()
        records = []
        for bits in product((0, 1), repeat=3):
            for i in range(125):
                name = "".join(map(str, bits)) + f"_{i}.png"
                (cls.data / "images" / name).touch()
                records.append(
                    dict(
                        filename=name,
                        bits=list(bits),
                        variation=i,
                        split="train" if i < 100 else "test",
                    )
                )
        (cls.data / "metadata.json").write_text(json.dumps(dict(records=records)))
        cls.manifests = {
            v: build(cls.data, v)
            for v in ("v1-positive-3rules", "v2-mixed-3rules", "v3-mixed-varied-rules")
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_truth_tables(self):
        tables = truth_tables(library("v3"))
        self.assertEqual(tables["R1"], [0, 1, 0, 1, 0, 1, 1, 1])
        self.assertEqual(tables["R4"], [0, 0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(len({tuple(t) for t in tables.values()}), 8)

    def test_split_and_generation(self):
        for version, m in self.manifests.items():
            self.assertEqual(m, build(self.data, version))
            self.assertEqual(
                validate(m, self.data)["partitions"],
                {"train": 800, "validation": 100, "test": 100},
            )
            self.assertTrue(any(s["images_reused_across_sets"] for s in m["sets"]))
            for item in m["sets"]:
                self.assertEqual(
                    consistent(item["observations"], m["rules"]), [item["rule"]]
                )
                self.assertEqual(len(item["observations"]), 10)
                if not version.startswith("v1"):
                    self.assertTrue(
                        all(
                            len(consistent([o], m["rules"])) > 1
                            for o in item["observations"]
                        )
                    )
                    self.assertEqual(sum(o["label"] for o in item["observations"]), 5)

    def test_reject_tampering_and_leakage(self):
        m = self.manifests["v3-mixed-varied-rules"]
        for kind in ("label", "duplicate", "split", "order", "reuse"):
            changed = copy.deepcopy(m)
            item = changed["sets"][0]
            if kind == "label":
                item["observations"][0]["label"] = not item["observations"][0]["label"]
            if kind == "duplicate":
                item["observations"][1] = item["observations"][0]
            if kind == "split":
                item["partition"] = "test"
            if kind == "order":
                item["option_order"][0] = item["option_order"][1]
            if kind == "reuse":
                item["images_reused_across_sets"] = not item[
                    "images_reused_across_sets"
                ]
            with self.assertRaises(ValueError, msg=kind):
                validate(changed, self.data)

    def test_options_shortcut_and_pairs(self):
        m = self.manifests["v3-mixed-varied-rules"]
        item = m["sets"][0]
        for k in range(8):
            scene = rotated(item, k)
            paths, text, mapping = prompt(scene, m, self.data)
            self.assertEqual(len(paths), 10)
            target = next(
                label for label, rule in mapping.items() if rule == item["rule"]
            )
            self.assertIn(f"{target}: {m['rules'][item['rule']]}", text)
        control, candidates = shortcut_control(item, m)
        self.assertIn(item["rule"], candidates)
        self.assertGreater(len(candidates), 1)
        pairs = make_pairs(m["sets"], "validation")
        self.assertEqual(len(pairs), 64)
        self.assertEqual(
            Counter(p["condition"] for p in pairs), {"same": 8, "different": 56}
        )

    def test_training_loop_checkpoint_and_resume_on_tiny_model(self):
        from unittest.mock import patch

        import torch
        import train_lora

        m = self.manifests["v3-mixed-varied-rules"]

        class Processor:
            def save_pretrained(self, path):
                Path(path).mkdir(exist_ok=True, parents=True)
                (Path(path) / "processor_test.json").write_text("{}")

        def fresh(*args, **kwargs):
            model = PeftTests().tiny()
            model.config._name_or_path = "Qwen/Qwen3.5-4B"
            return SimpleNamespace(
                model=model,
                processor=Processor(),
                tokenizer=SimpleNamespace(encode=lambda text, **kw: [ord(text) % 32]),
                _inputs=lambda paths, text: {"input_ids": torch.tensor([[1, 2, 3]])},
            )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "baseline"
            base.mkdir()
            atomic_json(
                base / "config.json",
                dict(
                    manifest_sha256=digest(m),
                    revision="main",
                    max_pixels=128 * 28 * 28,
                    partition="validation",
                ),
            )
            atomic_json(base / "metrics.json", dict(n=96))
            args = [
                "train_lora.py",
                "--manifest",
                "unused",
                "--data-dir",
                str(self.data),
                "--results-root",
                tmp,
                "--baseline-run",
                str(base),
                "--precision",
                "fp32",
                "--layers",
                "4",
                "--smoke-test",
                "--max-steps",
                "1",
                "--gradient-accumulation",
                "1",
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(train_lora, "load_manifest", return_value=m),
                patch.object(train_lora, "load", side_effect=fresh),
            ):
                train_lora.main()
            complete = list(Path(tmp).rglob("training_complete.json"))
            self.assertEqual(len(complete), 1)
            out = complete[0].parent
            checkpoint = out / "checkpoint-1"
            self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())
            self.assertTrue((checkpoint / "training_state.pt").is_file())
            self.assertTrue((out / "best_adapter/adapter_model.safetensors").is_file())
            # Exercise restoring optimizer, scaler and RNG from a full checkpoint boundary.
            complete[0].unlink()
            with (
                patch.object(sys, "argv", args + ["--resume", str(checkpoint)]),
                patch.object(train_lora, "load_manifest", return_value=m),
                patch.object(train_lora, "load", side_effect=fresh),
            ):
                train_lora.main()
            self.assertEqual(len(Journal(out / "losses.jsonl").rows), 1)


class RuntimeTests(unittest.TestCase):
    def test_results_resume_and_legacy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = {"task_version": "v3", "split_manifest": {}, "sets": []}
            out = prepare_run(tmp, 4, m, {"layers": [1]})
            self.assertEqual(out, prepare_run(tmp, 4, m, {"layers": [1]}))
            self.assertNotEqual(out, prepare_run(tmp, 4, m, {"layers": [2]}))
            journal = Journal(out / "rows.jsonl")
            journal.add("one", {"value": 2})
            self.assertEqual(len(Journal(out / "rows.jsonl").rows), 1)
            with self.assertRaises(ValueError):
                journal.add("one", {})
            self.assertEqual(
                str(LEGACY_PATCHING), "activation_patching/qwen3_5/dataset-v1"
            )

    def test_multitoken_likelihood_and_patch_cleanup(self):
        import torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()

            def forward(self, input_ids, logits_to_keep=1, **kwargs):
                hidden = self.layer(
                    torch.nn.functional.one_hot(input_ids, num_classes=4).float() * 4
                )
                return SimpleNamespace(logits=hidden[:, -logits_to_keep:])

        model = Toy()
        inputs = {
            "input_ids": torch.tensor([[0, 1]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        tokenizer = SimpleNamespace(
            encode=lambda x, **kwargs: {"a": [1], "b": [2, 2]}[x]
        )
        adapter = SimpleNamespace(model=model, tokenizer=tokenizer)
        a = answer_log_likelihood(model, inputs, [1])
        b = answer_log_likelihood(model, inputs, [2, 2])
        expected = torch.stack([a, b]).softmax(0).tolist()
        actual = score(adapter, inputs, {"a": "R1", "b": "R2"})
        self.assertAlmostEqual(actual["R1"], expected[0])
        before, vectors = capture(
            adapter, inputs, {"a": "R1", "b": "R2"}, [model.layer], "final"
        )
        after = patched(
            adapter,
            inputs,
            {"a": "R1", "b": "R2"},
            model.layer,
            torch.tensor([0.0, 0.0, 12.0, 0.0]),
            "final",
        )
        self.assertGreater(after["R2"], before["R2"])
        self.assertEqual(len(model.layer._forward_hooks), 0)
        with self.assertRaises(ValueError):
            patched(
                adapter,
                inputs,
                {"a": "R1", "b": "R2"},
                model.layer,
                torch.ones(3),
                "final",
            )
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_adapter_compatibility_and_probe_partitions(self):
        import numpy as np
        from probe_tasks import fit_probes
        from shared.model_qwen35 import validate_adapter

        with tempfile.TemporaryDirectory() as tmp:
            m = {"rules": ["R1", "R2"]}
            atomic_json(
                Path(tmp) / "pipeline.json",
                dict(manifest_sha256=digest(m), model_id="Qwen/Qwen3.5-4B"),
            )
            validate_adapter(tmp, m)
            with self.assertRaises(ValueError):
                validate_adapter(tmp, {"rules": ["different"]})
        items = [
            dict(
                partition="train" if i < 6 else "validation",
                rule="R1" if i % 2 else "R2",
            )
            for i in range(8)
        ]
        features = np.random.default_rng(3).normal(size=(8, 2, 3, 4))
        rows = fit_probes(features, items, "validation", 2026)
        self.assertEqual(len(rows), 12)
        self.assertTrue(
            all(
                r["train_size"] == 6
                and r["test_size"] == 2
                and r["majority_baseline"] == 0.5
                for r in rows
            )
        )

    def test_metrics_chance_and_orders(self):
        rows = [
            dict(
                set_id="a",
                truth="R1",
                correct=True,
                prediction="R1",
                control="intact",
                probabilities={"R1": 0.8, "R2": 0.2},
            ),
            dict(
                set_id="a",
                truth="R1",
                correct=False,
                prediction="R2",
                control="intact",
                probabilities={"R1": 0.2, "R2": 0.8},
            ),
        ]
        report = metrics(rows)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["chance"], 0.5)
        self.assertEqual(report["all_orders_correct"], 0)

    def test_matched_comparison_and_reject_incomplete(self):
        from shared.comparison import compare

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "baseline", root / "adapter"]
            plan = dict(
                pairs=[{"pair_id": "p"}],
                rotations=1,
                positions=["final"],
                layers=[1, 2],
            )
            for n, path in enumerate(paths):
                path.mkdir()
                atomic_json(
                    path / "config.json",
                    dict(
                        plan_hash=digest(plan),
                        plan=plan,
                        data_fingerprint="data",
                        model={"revision": "same"},
                        adapter="adapter" if n else None,
                    ),
                )
                journal = Journal(path / "patches.jsonl")
                for layer in (1, 2):
                    journal.add(
                        f"p:0:final:{layer}",
                        dict(
                            receiver="r",
                            rotation=0,
                            layer=layer,
                            condition="different",
                            position="final",
                            delta_receiver_probability=-0.1,
                            delta_donor_probability=0.1,
                            changed_answer=False,
                            receiver_correct=True,
                            donor_correct=True,
                        ),
                    )
            compare(paths, root / "comparison")
            self.assertTrue((root / "comparison/comparison.csv").is_file())
            (paths[1] / "patches.jsonl").write_text(
                (paths[1] / "patches.jsonl").read_text().splitlines()[0] + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Missing"):
                compare(paths, root / "incomplete")


class PeftTests(unittest.TestCase):
    def tiny(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(
            text_config=dict(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                rope_parameters={
                    "rope_type": "default",
                    "mrope_section": [1, 1, 2],
                    "partial_rotary_factor": 1.0,
                },
            ),
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=4,
                out_hidden_size=32,
                num_position_embeddings=16,
            ),
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=62,
            vision_end_token_id=63,
        )
        return Qwen3_5ForConditionalGeneration(config)

    def test_real_qwen35_peft_forward_freeze_save_reload(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        for layers in ([1], [4], None):
            torch.manual_seed(7)
            model = self.tiny()
            base_state = copy.deepcopy(model.state_dict())
            targets, selected = select_targets(model, layers)
            self.assertTrue(
                any("linear_attn" in t for t in targets)
                if layers != [4]
                else any("self_attn" in t for t in targets)
            )
            model = get_peft_model(
                model,
                LoraConfig(
                    r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM"
                ),
            )
            report = check_trainable(model, targets)
            self.assertGreater(report["count"], 0)
            inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
            loss = -answer_log_likelihood(model, inputs, [4])
            loss.backward()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.01
            )
            optimizer.step()
            for name, param in model.base_model.model.named_parameters():
                if "lora_" not in name:
                    original = name.replace(".base_layer", "")
                    self.assertTrue(
                        torch.equal(param.detach(), base_state[original]), name
                    )
            with tempfile.TemporaryDirectory() as tmp:
                model.eval()
                expected = model(**inputs, use_cache=False).logits.detach()
                model.save_pretrained(tmp)
                fresh = self.tiny()
                fresh.load_state_dict(base_state)
                restored = PeftModel.from_pretrained(fresh, tmp).eval()
                self.assertTrue(
                    torch.allclose(
                        expected, restored(**inputs, use_cache=False).logits, atol=1e-6
                    )
                )
            with self.assertRaises(ValueError):
                select_targets(self.tiny(), [5])


if __name__ == "__main__":
    unittest.main()
