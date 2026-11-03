"""CPU checks for the patching plan and the hidden-state replacement hook.

Run: python three-operand-experiments/code/03-activation-patching/test_patching.py
"""
import itertools
import unittest
from types import SimpleNamespace

import torch

import operator_controls_qwen35 as exp
from patch_activations_qwen35 import (
    check_sets, make_pairs, patched_probabilities, record_original,
)
from probe_rules import make_sets


def fake_records():
    rows = []
    for bits in itertools.product((0, 1), repeat=3):
        for variation in range(125):
            rows.append({
                "filename": f'scene_{"".join(map(str, bits))}_{variation:03d}.png',
                "bits": list(bits),
                "variation": variation,
                "split": "train" if variation < 100 else "test",
            })
    return rows


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, input_ids, logits_to_keep=1, use_cache=False):
        hidden = self.layer(input_ids.float())
        return SimpleNamespace(logits=hidden[:, -1:, :])


class PatchingTests(unittest.TestCase):
    def test_pairs_have_no_shared_images(self):
        experiment = SimpleNamespace(
            records=fake_records(), RULES=exp.RULES,
            satisfies=exp.satisfies, compatible_rules=exp.compatible_rules,
        )
        sets = make_sets(experiment)
        self.assertEqual(len(sets), 39)
        pairs = make_pairs(sets, 5)
        report = check_sets(sets, pairs, experiment.records)
        self.assertEqual(report["same_rule_pairs"], 15)
        self.assertEqual(report["different_rule_pairs"], 30)
        for pair in pairs:
            self.assertEqual(
                pair["condition"] == "same",
                pair["receiver_rule"] == pair["donor_rule"],
            )

    def test_hook_changes_only_the_selected_position(self):
        model = SimpleModel()
        adapter = SimpleNamespace(model=model)
        mapping = {"X": "R1", "Y": "R2", "Z": "R3"}
        original_ids = exp.TOKEN_IDS
        try:
            exp.TOKEN_IDS = {"X": 0, "Y": 1, "Z": 2}
            inputs = {"input_ids": torch.tensor(
                [[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            )}
            before = record_original(adapter, inputs, mapping, [model.layer])
            self.assertEqual(before["prediction"], "R1")
            donor = torch.tensor([0.0, 12.0, 0.0])
            last = patched_probabilities(
                adapter, inputs, mapping, model.layer, donor, "final"
            )
            previous = patched_probabilities(
                adapter, inputs, mapping, model.layer, donor, "previous"
            )
            self.assertGreater(last["R2"], 0.99)
            self.assertAlmostEqual(
                previous["R2"], before["probabilities"]["R2"], places=6
            )
            again = record_original(adapter, inputs, mapping, [model.layer])
            self.assertEqual(again["prediction"], "R1")
        finally:
            exp.TOKEN_IDS = original_ids


if __name__ == "__main__":
    unittest.main()
