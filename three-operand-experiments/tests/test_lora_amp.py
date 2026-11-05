"""Exercise real GradScaler skip/update behaviour on CPU, without model weights."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/04-lora"))
from train_lora import optimizer_update, check_smoke_updates


class AMPTests(unittest.TestCase):
    def model(self):
        import torch

        model = torch.nn.Module()
        model.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(2, 2)))
        return model

    def test_finite_gradient_runs_adamw_and_changes_b(self):
        import torch

        model = self.model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
        scaler.scale(model.lora_B.sum()).backward()
        report = optimizer_update(model, opt, scaler)
        self.assertTrue(report["optimizer_step_occurred"])
        self.assertTrue(report["lora_b_weights_changed"])
        self.assertEqual(report["nonfinite_gradient_values"], 0)
        self.assertEqual(report["optimizer_state_entries"], 1)
        self.assertEqual(report["gradient_scale_before"], 16.0)
        check_smoke_updates([report])

    def test_overflow_is_skipped_logged_and_fails_smoke(self):
        import torch

        model = self.model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
        reports = []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            scaler.scale(model.lora_B.sum()).backward()
            model.lora_B.grad.fill_(float("inf"))
            reports.append(optimizer_update(model, opt, scaler))
        self.assertEqual(reports[0]["nonfinite_gradient_values"], 4)
        self.assertEqual(reports[0]["nonfinite_gradient_parameters"], ["lora_B"])
        self.assertEqual(reports[-1]["gradient_scale_after"], 4.0)
        self.assertFalse(reports[-1]["optimizer_step_occurred"])
        self.assertFalse(reports[-1]["lora_b_weights_changed"])
        self.assertEqual(len(opt.state), 0)
        with self.assertRaisesRegex(RuntimeError, "Smoke test failed"):
            check_smoke_updates(reports)
        # The same optimizer recovers once gradients become finite.
        opt.zero_grad(set_to_none=True)
        scaler.scale(model.lora_B.sum()).backward()
        recovered = optimizer_update(model, opt, scaler)
        check_smoke_updates(reports + [recovered])
        self.assertTrue(recovered["optimizer_step_occurred"])

    def test_unscaled_nonfinite_gradients_are_not_applied(self):
        import torch

        model = self.model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        model.lora_B.sum().backward()
        model.lora_B.grad.fill_(float("nan"))
        report = optimizer_update(model, opt, scaler)
        self.assertFalse(report["optimizer_step_occurred"])
        self.assertEqual(len(opt.state), 0)


if __name__ == "__main__":
    unittest.main()
