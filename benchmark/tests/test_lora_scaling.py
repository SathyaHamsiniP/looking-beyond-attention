"""CPU design/integration tests. No pretrained loading or GPU training."""
import setup_paths
from benchmark_paths import BENCHMARK, source_path
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lora_scaling as scaling


def fixture():
    sets=[]
    for split,n in [('train',30),('validation',10),('test',10)]:
        for rule in scaling.RULES:
            for i in range(n):
                sets.append(dict(set_id=f'{split}_{rule}_{i}',partition=split,rule=rule,
                    image_files=[f'{split}_{rule}_{i}_{j}.png' for j in range(10)],
                    option_mapping=dict(zip('XYZ',scaling.RULES[i%3:]+scaling.RULES[:i%3]))))
    return dict(sets=sets)


class ScalingTests(unittest.TestCase):
    def plan(self, manifest=None, **kwargs):
        manifest=fixture() if manifest is None else manifest
        with patch.object(scaling,'MANIFEST_HASH',scaling.digest(manifest)):
            return scaling.make_plan(manifest,**kwargs)

    def test_balanced_nested_reproducible_and_fixed_heldout(self):
        m=fixture();before=copy.deepcopy(m)
        plan=self.plan(m)
        ids=[]
        for n in scaling.BUDGETS:
            c=plan['conditions'][str(n)]
            self.assertEqual(c['sets'],n)
            self.assertEqual(c['sets_per_rule'],{r:n//3 for r in scaling.RULES})
            self.assertEqual(c['answer_label_counts'],{l:n//3 for l in 'XYZ'})
            ids.append(set(c['set_ids']))
        self.assertLess(ids[0],ids[1]);self.assertLess(ids[1],ids[2])
        self.assertEqual(m,before)
        self.assertEqual(plan,self.plan(m))
        for split in ('validation','test'):
            self.assertEqual(plan[split]['selected_sets_hash'],scaling.digest([s for s in m['sets'] if s['partition']==split]))
        self.assertEqual(plan['conditions']['90']['selected_sets_hash'],scaling.digest([s for s in m['sets'] if s['partition']=='train']))

    def test_fixed_successful_steps_full_batches_and_passes(self):
        plan=self.plan()
        reference=plan['conditions']['24']['training_config']
        for n in scaling.BUDGETS:
            c=plan['conditions'][str(n)]
            self.assertEqual(c['training_config'],reference)
            self.assertEqual(len(c['update_schedule']),46)
            self.assertTrue(all(len(batch)==4 for batch in c['update_schedule']))
            self.assertTrue(all(sid in c['set_ids'] for batch in c['update_schedule'] for sid in batch))
            self.assertAlmostEqual(c['effective_passes'],184/n)
        self.assertFalse(plan['previous_100_percent']['directly_comparable'])
        self.assertEqual(plan['previous_100_percent']['recorded_updates'],46)

    def test_seeds_change_schedule_not_heldout_or_subset_membership(self):
        a,b=self.plan(seed=2026),self.plan(seed=2027)
        self.assertEqual(a['validation'],b['validation']);self.assertEqual(a['test'],b['test'])
        for n in scaling.BUDGETS:
            self.assertEqual(a['conditions'][str(n)]['set_ids'],b['conditions'][str(n)]['set_ids'])
            self.assertNotEqual(a['conditions'][str(n)]['update_schedule'],b['conditions'][str(n)]['update_schedule'])

    def test_leakage_duplicate_ids_and_wrong_manifest_rejected(self):
        m=fixture();m['sets'][-1]['image_files'][0]=m['sets'][0]['image_files'][0]
        with self.assertRaisesRegex(ValueError,'leakage'):
            self.plan(m)
        m=fixture();m['sets'][-1]['set_id']=m['sets'][0]['set_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.plan(m)
        with self.assertRaisesRegex(ValueError,'exact completed'):
            scaling.make_plan(fixture())

    def test_output_isolation_resume_config_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[scaling.output_path(tmp,2026,c) for c in ('pretrained','24','45','90')]
            self.assertEqual(len(set(paths)),4)
            self.assertNotEqual(paths[1],scaling.output_path(tmp,2026,'24',True))
            cfg={'x':1}
            with scaling.locked_output(paths[0],cfg,False):
                self.assertEqual(scaling.read(paths[0]/'config.json'),cfg)
                with self.assertRaisesRegex(ValueError,'Another process'):
                    with scaling.locked_output(paths[0],cfg,True):
                        pass
            with self.assertRaises(ValueError):
                with scaling.locked_output(paths[0],cfg,False):
                    pass
            with self.assertRaisesRegex(ValueError,'differs'):
                with scaling.locked_output(paths[0],{'x':2},True):
                    pass
            with scaling.locked_output(paths[0],cfg,True):
                pass

    def test_rotation_metrics_and_paired_input_aggregation(self):
        rows=[]
        for rule in scaling.RULES:
            for rot in range(3):
                pred=scaling.RULES[rot]
                rows.append(dict(set_id=rule,true_rule=rule,rotation=rot,prediction=pred,
                                 answer_label='Y',correct=pred==rule))
        report=scaling.metrics(rows)
        self.assertEqual(report['sets'],3)
        self.assertAlmostEqual(report['accuracy'],1/3)
        self.assertEqual(report['rotation_consistency'],0)
        self.assertEqual(report['answer_labels']['Y'],9)
        with self.assertRaises(ValueError):
            scaling.metrics(rows[:-1])

    def test_cpu_comparison_outputs_and_incompatible_runtime_rejected(self):
        plan=self.plan()
        m=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for condition in ('pretrained','24','45','90'):
                path=scaling.output_path(tmp,2026,condition);path.mkdir(parents=True)
                config=dict(plan_hash=scaling.digest(plan),runner_sha256='same',software={'torch':'fixture'},
                    model_settings={'revision':scaling.REVISION},source_commit=scaling.SOURCE_COMMIT,
                    manifest_hash='fixture',seed=2026,smoke_test=False,partition='validation',
                    training_config=None if condition=='pretrained' else plan['conditions'][condition]['training_config'])
                scaling.write(path/'config.json',config)
                if condition!='pretrained':
                    scaling.write(path/'complete.json',dict(successful_optimizer_steps=46))
                    scaling.write(path/'trainable.json',dict(target_modules=['q_proj']))
                    losses=[dict(step=i,training_loss=1.,optimizer_step_occurred=True,
                        lora_b_weights_changed=True,lora_b_matrices_changed=248,
                        **({'validation_loss':1.} if i%10==0 or i==46 else {})) for i in range(1,47)]
                    (path/'losses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in losses))
                rows=[]
                for item in (s for s in m['sets'] if s['partition']=='validation'):
                    for rotation in range(3):
                        order=list(scaling.RULES[rotation:]+scaling.RULES[:rotation])
                        mapping=dict(zip('XYZ',order));pred=mapping['Y']
                        rows.append(dict(set_id=item['set_id'],true_rule=item['rule'],rotation=rotation,
                            image_files=item['image_files'],mapping=mapping,prediction=pred,answer_label='Y',
                            probabilities={r:.8 if r==pred else .1 for r in scaling.RULES},correct=pred==item['rule']))
                (path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):
                scaling.compare_outputs(tmp,2026,plan)
            comparison=Path(tmp)/'seed-2026/comparison'
            self.assertTrue((comparison/'loss_curves.png').is_file())
            self.assertAlmostEqual(scaling.read(comparison/'summary.json')['90']['accuracy'],1/3)
            config_path=scaling.output_path(tmp,2026,'45')/'config.json'
            config=scaling.read(config_path);config['software']['torch']='different'
            scaling.write(config_path,config)
            with self.assertRaisesRegex(ValueError,'Software'):
                scaling.compare_outputs(tmp,2026,plan)

    def test_selected_peft_weights_round_trip_cpu(self):
        # Check the exact PEFT key format used by selected-weight recovery.
        import torch
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        from transformers import LlamaConfig,LlamaForCausalLM
        model=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=8,
            intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)),
            LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],task_type='CAUSAL_LM'))
        weights={k:v.detach().clone() for k,v in get_peft_model_state_dict(model).items()}
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_' in name:
                    p.add_(1)
        set_peft_model_state_dict(model,weights)
        self.assertTrue(all(torch.equal(v,weights[k]) for k,v in get_peft_model_state_dict(model).items()))


if __name__=='__main__':
    unittest.main()
