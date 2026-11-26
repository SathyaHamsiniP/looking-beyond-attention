import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/10-layer-patching'))
import layer_patching as experiment


class LayerPatchingTests(unittest.TestCase):
    def test_restore_only_selected_modules_on_failure(self):
        a, b = [SimpleNamespace(scaling={'default': scale}) for scale in (2., 3.)]
        with self.assertRaises(RuntimeError):
            with experiment.silence([a]):
                self.assertEqual((a.scaling['default'], b.scaling['default']), (0., 3.))
                raise RuntimeError()
        self.assertEqual(a.scaling, {'default': 2.})

    def test_selection_averages_within_sets(self):
        rows = []
        for sid, delta in [('a', 3), ('a', 3), ('b', 1)]:
            rows.append(dict(set_id=sid, true_rule='R4', loss_increases=[0]*30+[delta, 1]))
        ranking = experiment.rank_layers(rows)
        self.assertEqual((ranking[0]['layer'], ranking[0]['mean_loss_increase']), (31, 2))

    def test_paired_effect_and_disjoint_subset(self):
        variants = [{}, {}]
        for rule in experiment.choices.IDS:
            for condition in ['same','different']:
                for i, variant in enumerate(variants):
                    value = (0.2 if condition == 'different' else 0.05)*i
                    variant[rule, condition] = dict(receiver_set=rule, receiver_rule=rule,
                        rotation=0, condition=condition, delta_true_rule=value, delta_donor_answer=0,
                        donor_predicted_rule='R1', donor_true_rule='R2', donor_rule='R2',
                        shared_images=int(rule=='R4'))
        result = experiment.compare(*variants)
        self.assertAlmostEqual(result['mean_control_adjusted_lora_minus_base'], .15)
        self.assertEqual(result['disjoint_receiver_count'], 6)
        self.assertAlmostEqual(result['disjoint_control_adjusted_mean'], .15)
