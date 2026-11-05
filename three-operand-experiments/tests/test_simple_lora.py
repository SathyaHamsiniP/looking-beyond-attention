"""CPU checks for the small three-rule LoRA experiment. No pretrained weights."""

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE / "04-lora"))
import build_training_sets as data
import common
import train_lora


def tiny_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    cfg = Qwen3_5Config(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
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
    return Qwen3_5ForConditionalGeneration(cfg)


class SimpleLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "images").mkdir()
        records = []
        for bits in data.BITS:
            for index in range(125):
                name = "".join(map(str, bits)) + f"_{index}.png"
                (cls.root / "images" / name).touch()
                records.append(
                    dict(
                        filename=name,
                        bits=list(bits),
                        split="train" if index < 100 else "test",
                    )
                )
        (cls.root / "metadata.json").write_text(json.dumps(dict(records=records)))
        cls.manifest = data.build(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_counts_and_no_leakage(self):
        m = self.manifest
        self.assertEqual(m, data.build(self.root))
        self.assertEqual(
            Counter(m["image_partitions"].values()),
            dict(train=800, validation=100, test=100),
        )
        self.assertEqual(
            Counter(s["partition"] for s in m["sets"]),
            dict(train=90, validation=30, test=30),
        )
        identities = set()
        for s in m["sets"]:
            self.assertEqual(len(s["image_files"]), 10)
            self.assertEqual(len(set(s["image_files"])), 10)
            identity = (s["partition"], tuple(sorted(s["image_files"])))
            self.assertNotIn(identity, identities)
            identities.add(identity)
            self.assertTrue(
                all(
                    m["image_partitions"][f] == s["partition"] for f in s["image_files"]
                )
            )
            self.assertEqual(
                {tuple(b) for b in s["bits"]},
                {b for b in data.BITS if data.satisfies(s["rule"], b)},
            )
            self.assertEqual(
                [r for r in data.RULES if all(data.satisfies(r, b) for b in s["bits"])],
                [s["rule"]],
            )
        for partition, count in data.COUNTS.items():
            self.assertEqual(
                Counter(s["rule"] for s in m["sets"] if s["partition"] == partition),
                {r: count for r in data.RULES},
            )
        self.assertTrue(any(s["reused_image_files"] for s in m["sets"]))

    def test_manifest_rejects_changed_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            m = copy.deepcopy(self.manifest)
            m["sets"][0]["rule"] = "R2"
            path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                data.load(path, self.root)
        for s in self.manifest["sets"]:
            self.assertEqual(s["option_mapping"][s["target_label"]], s["rule"])
        for rule in data.RULES:
            self.assertEqual(
                Counter(
                    s["target_label"]
                    for s in self.manifest["sets"]
                    if s["partition"] == "train" and s["rule"] == rule
                ),
                dict(X=10, Y=10, Z=10),
            )

    def test_mismatched_example_settings(self):
        with self.assertRaises(ValueError):
            common.matched({"set_id": "x"}, {"set_id": "y"}, ("set_id",))

    def test_multitoken_loss_alignment(self):
        import torch

        class Toy(torch.nn.Module):
            def forward(self, input_ids, logits_to_keep=1, **kwargs):
                return SimpleNamespace(
                    logits=torch.nn.functional.one_hot(
                        input_ids, num_classes=5
                    ).float()[:, -logits_to_keep:]
                    * 3
                )

        ids = torch.tensor([[1, 2, 3]])
        model = Toy()
        actual = common.answer_logp(model, {"input_ids": ids}, [4, 2])
        scores = torch.stack(
            [
                torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0]).log_softmax(0)[4],
                torch.tensor([0.0, 0.0, 0.0, 0.0, 3.0]).log_softmax(0)[2],
            ]
        ).sum()
        self.assertTrue(torch.allclose(actual, scores))

    def test_peft_freeze_and_adapter_reload(self):
        import torch
        from peft import LoraConfig, get_peft_model, PeftModel

        model = tiny_model()
        original = copy.deepcopy(model.state_dict())
        selected = train_lora.targets(model)
        self.assertTrue(any("linear_attn" in t for t in selected))
        self.assertTrue(any("self_attn" in t for t in selected))
        self.assertFalse(any("visual" in t for t in selected))
        model = get_peft_model(
            model,
            LoraConfig(
                r=2, lora_alpha=4, target_modules=selected, task_type="CAUSAL_LM"
            ),
        )
        train_lora.trainable(model, selected)
        loss = -common.answer_logp(model, {"input_ids": torch.tensor([[1, 2, 3]])}, [4])
        loss.backward()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        opt.step()
        for name, param in model.base_model.model.named_parameters():
            if "lora_" not in name:
                self.assertTrue(
                    torch.equal(
                        param.detach(), original[name.replace(".base_layer", "")]
                    ),
                    name,
                )
        model.eval()
        sample = {"input_ids": torch.tensor([[1, 2, 3]])}
        expected = model(**sample, use_cache=False).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            fresh = tiny_model()
            fresh.load_state_dict(original)
            restored = PeftModel.from_pretrained(fresh, tmp).eval()
            self.assertTrue(
                torch.allclose(
                    expected, restored(**sample, use_cache=False).logits, atol=1e-6
                )
            )

    def test_training_smoke_checkpoint_resume_cpu(self):
        import torch

        class Processor:
            def save_pretrained(self, path):
                Path(path).mkdir(exist_ok=True, parents=True)

        def fresh(*args, **kwargs):
            model = tiny_model()
            model.config._name_or_path = common.MODEL
            return SimpleNamespace(
                model=model,
                processor=Processor(),
                tokenizer=SimpleNamespace(encode=lambda text, **kw: [ord(text) % 32]),
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            args = [
                "train_lora.py",
                "--revision",
                "a" * 40,
                "--precision",
                "fp32",
                "--manifest",
                str(path),
                "--data-dir",
                str(self.root),
                "--output-root",
                str(Path(tmp) / "runs"),
                "--smoke-test",
                "--rank",
                "2",
                "--checkpoint-interval",
                "1",
            ]
            prepared = lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])}
            with (
                patch.object(sys, "argv", args),
                patch.object(train_lora, "load_model", side_effect=fresh),
                patch.object(train_lora, "inputs", side_effect=prepared),
            ):
                train_lora.main()
            complete = next((Path(tmp) / "runs").rglob("complete.json"))
            out = complete.parent
            self.assertTrue((out / "best_adapter/adapter_model.safetensors").is_file())
            self.assertTrue((out / "checkpoint-1/training_state.pt").is_file())
            # Simulate recovery after checkpoint 2 was written, before the completion marker.
            complete.unlink()
            with (
                patch.object(
                    sys, "argv", args + ["--resume", str(out / "checkpoint-2")]
                ),
                patch.object(train_lora, "load_model", side_effect=fresh),
                patch.object(train_lora, "inputs", side_effect=prepared),
            ):
                train_lora.main()
            self.assertEqual(len(common.rows(out / "losses.jsonl")), 2)

    def test_evaluation_and_prediction_comparison_without_weights(self):
        import evaluate_lora
        import compare_predictions

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(self.manifest))
            opts = [
                "--revision",
                "a" * 40,
                "--precision",
                "fp32",
                "--manifest",
                str(manifest),
                "--data-dir",
                str(self.root),
                "--output-root",
                str(root / "evaluation"),
            ]

            def scores(adapter, prepared, rotation):
                return {r: 0.8 if r == prepared["rule"] else 0.1 for r in data.RULES}

            with (
                patch.object(sys, "argv", ["evaluate_lora.py", *opts]),
                patch.object(evaluate_lora, "load_model", return_value=object()),
                patch.object(
                    evaluate_lora,
                    "inputs",
                    side_effect=lambda adapter, item, *args: {"rule": item["rule"]},
                ),
                patch.object(evaluate_lora, "score", side_effect=scores),
            ):
                evaluate_lora.main()
            before = next((root / "evaluation").iterdir())
            adapter = root / "adapter"
            adapter.mkdir()
            args = SimpleNamespace(
                revision="a" * 40, precision="fp32", max_pixels=128 * 28 * 28
            )
            common.save(
                adapter / "experiment.json",
                dict(
                    settings=common.settings(args),
                    manifest_hash=common.digest(self.manifest),
                ),
            )
            with (
                patch.object(
                    sys, "argv", ["evaluate_lora.py", *opts, "--adapter", str(adapter)]
                ),
                patch.object(evaluate_lora, "load_model", return_value=object()),
                patch.object(
                    evaluate_lora,
                    "inputs",
                    side_effect=lambda adapter, item, *args: {"rule": item["rule"]},
                ),
                patch.object(evaluate_lora, "score", side_effect=scores),
            ):
                evaluate_lora.main()
            after = next(p for p in (root / "evaluation").iterdir() if p != before)
            compare_args = [
                "compare_predictions.py",
                "--baseline",
                str(before),
                "--finetuned",
                str(after),
                "--output",
                str(root / "comparison"),
            ]
            with patch.object(sys, "argv", compare_args):
                compare_predictions.main()
            result = json.loads((root / "comparison/comparison.json").read_text())
            self.assertEqual(result["before"]["trials"], 90)
            self.assertEqual(result["accuracy_difference"], 0)
            rowfile = after / "predictions.jsonl"
            all_rows = rowfile.read_text().splitlines()
            rowfile.write_text("\n".join(all_rows[:-1]) + "\n")
            with patch.object(sys, "argv", compare_args), self.assertRaises(ValueError):
                compare_predictions.main()



if __name__ == "__main__":
    unittest.main()
