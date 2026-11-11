"""CPU fixtures for seven-rule data, readout, gradients and checkpoint safety.

No pretrained Qwen weights, 4-bit GPU execution or real GPU training in tests.
"""
import setup_paths
from benchmark_paths import BENCHMARK, source_path
from collections import Counter
import copy
from contextlib import redirect_stdout
import io
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lora_7rules as experiment


class SevenRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.data=cls.root/'data';(cls.data/'images').mkdir(parents=True)
        records=[]
        from PIL import Image
        image=io.BytesIO();Image.new('RGB',(8,8),color='green').save(image,format='PNG')
        for bits in product((0,1),repeat=3):
            for i in range(125):
                name=f"scene_{''.join(map(str,bits))}_{i:03d}.png"
                (cls.data/'images'/name).write_bytes(image.getvalue())
                records.append(dict(filename=name,bits=list(bits),split='train' if i<100 else 'test'))
        (cls.data/'metadata.json').write_text(json.dumps(dict(records=records)))
        cls.manifest,cls.mapping_bytes=experiment.generate(cls.data)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_truth_tables_independent_oracle(self):
        expected=['001 011 101 110 111','010 011 101 110 111','011 100 101 110 111',
            '001 010 011 100 101 111','001 010 011 100 110 111','001 010 100 101 110 111',
            '001 010 011 100 101 110']
        rules=self.manifest['rules']['rules']
        self.assertEqual([r['true_bits'] for r in rules],[s.split() for s in expected])
        self.assertEqual(len({tuple(r['true_bits']) for r in rules}),7)

    def test_partition_sizes_strata_and_no_leakage(self):
        m=self.manifest;assign=m['image_partitions']
        self.assertEqual(Counter(assign.values()),dict(train=800,validation=100,test=100))
        used={s:set() for s in experiment.COUNTS}
        for item in m['sets']:
            used[item['split']].update(item['image_files'])
            self.assertTrue(all(assign[name]==item['split'] for name in item['image_files']))
        self.assertTrue(all(not used[a]&used[b] for a in used for b in used if a!=b))
        records=experiment.read(self.data/'metadata.json')['records']
        report=experiment.validate_partitions(records,assign)
        self.assertEqual(len(report),8)
        bad=assign.copy();bad[next(k for k,v in bad.items() if v=='train')]='test'
        with self.assertRaises(ValueError):experiment.validate_partitions(records,bad)

    def test_exact_set_counts_coverage_and_identifiability(self):
        m=self.manifest;rules=m['rules']['rules'];identities=set()
        self.assertEqual(len(m['sets']),294)
        self.assertEqual(Counter(s['split'] for s in m['sets']),dict(train=147,validation=49,test=98))
        for item in m['sets']:
            self.assertEqual(len(item['image_files']),10)
            self.assertEqual(len(set(item['image_files'])),10)
            table=next(r['true_bits'] for r in rules if r['id']==item['rule'])
            self.assertEqual(set(item['bits']),set(table))
            self.assertEqual(experiment.builder.compatible_rules(item['bits'],rules),[item['rule']])
            identity=tuple(sorted(item['image_files']));self.assertNotIn(identity,identities);identities.add(identity)
        for split,n in experiment.COUNTS.items():
            self.assertEqual(Counter(s['rule'] for s in m['sets'] if s['split']==split),{r:n for r in experiment.comparison.IDS})

    def test_deterministic_generation_and_answer_balancing(self):
        again,mapping=experiment.generate(self.data)
        self.assertEqual(self.manifest,again);self.assertEqual(self.mapping_bytes,mapping)
        for split,n in experiment.COUNTS.items():
            for rule in experiment.comparison.IDS:
                items=[s for s in again['sets'] if s['split']==split and s['rule']==rule]
                self.assertEqual(Counter(s['target_label'] for s in items),{str(i):n//7 for i in range(1,8)})
                for s in items:
                    self.assertEqual(s['option_mapping'][s['target_label']],rule)
                    self.assertEqual(s['option_mapping'],experiment.comparison.mapping(7,s['fixed_rotation']))
        jobs=experiment.training_schedule(again)
        self.assertEqual(len(jobs),111)
        self.assertEqual(sum(len(j['set_ids']) for j in jobs),441)
        for epoch in range(1,4):
            sub=[j for j in jobs if j['epoch']==epoch]
            self.assertEqual(len(sub),37);self.assertEqual(len(sub[-1]['set_ids']),3)
            self.assertEqual(len({sid for j in sub for sid in j['set_ids']}),147)
            self.assertEqual(sum(j['epoch_end'] for j in sub),1)

    def test_validation_refuses_tampered_sets(self):
        for field in ('duplicate','bits','split','mapping','coverage'):
            m=copy.deepcopy(self.manifest);s=m['sets'][0]
            if field=='duplicate':m['sets'][1]=copy.deepcopy(s)
            elif field=='bits':s['bits'][0]='000'
            elif field=='split':s['split']='test'
            elif field=='mapping':s['target_label']='7' if s['target_label']!='7' else '1'
            else:
                # Keep filenames/metadata paired but remove a required configuration.
                bits=s['bits'][0]
                replacement=next(x for x in m['sets'][2]['image_files'] if x not in s['image_files'] and x[6:9]!=bits)
                s['image_files'][0]=replacement;s['bits'][0]=replacement[6:9]
            with self.subTest(field=field),self.assertRaises(ValueError):experiment.validate_manifest(m,self.data)

    def test_historical_mapping_identity_and_reconstruction_provenance(self):
        self.assertIn('reconstructed',self.manifest['provenance']['method'])
        _,identity=experiment.dataset_identity(self.data)
        history=dict(dataset_fingerprint=identity,image_partitions=self.manifest['image_partitions'])
        path=self.root/'fixture-historical.json';experiment.atomic_json(path,history)
        with patch.object(experiment,'HISTORICAL_HASH',experiment.digest(history)):
            mapping,provenance=experiment.partitions(self.data,path)
            self.assertEqual(mapping,self.manifest['image_partitions'])
            self.assertIn('verified exact',provenance['method'])
        with self.assertRaisesRegex(ValueError,'hash'):experiment.partitions(self.data,path)

    def test_numeric_tokenization_boundary_and_preprocessing_parity(self):
        import torch
        class Tokenizer:
            def encode(self,text,**kwargs):
                return [int(text)] if text.isdigit() else [10]+([int(text[-1])] if text[-1].isdigit() else [])
        class Prepared(dict):
            def to(self,device):return self
        class Processor:
            tokenizer=Tokenizer()
            def __init__(self):self.calls=[]
            def apply_chat_template(self,messages,**kwargs):
                self.calls.append((messages,kwargs))
                if kwargs['tokenize'] is False:return 'assistant:'
                return Prepared(input_ids=torch.tensor([[10,11]]))
        class Model:
            device='cpu'
            def __call__(self,**kwargs):return SimpleNamespace(logits=torch.arange(20,dtype=torch.float32).reshape(1,1,20))
        processor=Processor();model=Model();item=self.manifest['sets'][0]
        tokens=experiment.comparison.label_tokens(processor.tokenizer)
        experiment.prepare(model,processor,item,self.data,'prompt',tokens)
        ours=processor.calls[-1]
        experiment.comparison.evaluate_numeric(model,processor,tokens,item['image_files'],self.data,'prompt',100352)
        original=processor.calls[-1]
        self.assertEqual(ours[1],original[1])
        self.assertEqual([c['image'].tobytes() for c in ours[0][0]['content'][:-1]],
                         [c['image'].tobytes() for c in original[0][0]['content'][:-1]])
        with patch.object(Tokenizer,'encode',return_value=[1,2]):
            with self.assertRaises(ValueError):experiment.comparison.label_tokens(processor.tokenizer)
        with patch.object(Tokenizer,'encode',side_effect=lambda text,**kw:[10]):
            with self.assertRaisesRegex(ValueError,'boundary'):experiment.prepare(model,processor,item,self.data,'prompt',tokens)

    def test_answer_loss_and_partial_accumulation_gradient(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.arange(12,dtype=torch.float32)/12)
            def forward(self,**kwargs):return SimpleNamespace(logits=self.values.reshape(1,1,-1))
        model=Model()
        loss=experiment.answer_loss(model,{},3)
        self.assertTrue(torch.allclose(loss,-model.values.log_softmax(0)[3]))
        loss.backward();self.assertLess(float(model.values.grad[3]),0)
        self.assertTrue(bool(torch.isfinite(model.values.grad).all()))
        # The last update has three examples, normalized by three rather than four.
        targets=[1,2,3];model.zero_grad()
        for target in targets:(experiment.answer_loss(model,{},target)/len(targets)).backward()
        actual=model.values.grad.clone();model.zero_grad()
        torch.stack([experiment.answer_loss(model,{},target) for target in targets]).mean().backward()
        self.assertTrue(torch.allclose(actual,model.values.grad))

    def tiny_adapter(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from peft import LoraConfig,get_peft_model
        import torch
        model=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
        model.vision_tower=torch.nn.Linear(8,8)
        targets=experiment.decoder_targets(model)
        self.assertFalse(any('vision' in name for name in targets))
        for p in model.parameters():p.requires_grad=False
        model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=targets,task_type='CAUSAL_LM'))
        return model,targets

    def test_decoder_freezing_real_cpu_optimizer_and_weight_changes(self):
        import torch
        model,targets=self.tiny_adapter();inventory=experiment.trainable_inventory(model,targets)
        self.assertEqual(inventory['target_count'],len(targets))
        self.assertTrue(all('lora_' in n for n,p in model.named_parameters() if p.requires_grad))
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        experiment.answer_loss(model,dict(input_ids=torch.tensor([[1,2,3]])),4).backward()
        evidence=experiment.optimizer_update(model,optimizer,scaler)
        self.assertTrue(evidence['optimizer_step_occurred']);self.assertTrue(evidence['finite_gradients'])
        self.assertGreater(evidence['lora_b_matrices_changed'],0)
        self.assertTrue(all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen))

    def test_bad_gradients_not_counted_successful(self):
        import torch
        model,_=self.tiny_adapter()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.full_like(p,float('inf'))
        evidence=experiment.optimizer_update(model,optimizer,scaler)
        self.assertFalse(evidence['optimizer_step_occurred']);self.assertFalse(evidence['finite_gradients'])
        self.assertEqual(evidence['lora_b_matrices_changed'],0)

    def test_checkpoint_roundtrip_rng_journal_and_output_refusals(self):
        import torch
        model,_=self.tiny_adapter()
        class Processor:
            def save_pretrained(self,path):experiment.atomic_json(Path(path)/'processor.json',{})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'training';config={'manifest':'one','model':experiment.MODEL}
            with experiment.output(path,config):
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
                state=dict(config_hash=experiment.digest(config),next_update=1,
                    losses=[dict(update=1,optimizer_step_occurred=True)],rng=torch.get_rng_state(),optimizer=optimizer.state_dict())
                experiment.save_checkpoint(path,model,Processor(),state)
                recovered=experiment.checkpoint_state(path,experiment.digest(config))
                self.assertTrue(torch.equal(recovered['rng'],state['rng']))
                self.assertEqual(recovered['next_update'],1)
                with self.assertRaisesRegex(ValueError,'Another process'):
                    with experiment.output(path,config,True):pass
            with self.assertRaises(ValueError):
                with experiment.output(path,config):pass
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                with experiment.output(path,{'manifest':'other'},True):pass
            with self.assertRaisesRegex(ValueError,'mismatch'):experiment.checkpoint_state(path,'other')
            # Recover the previous complete checkpoint after interrupted directory replacement.
            (path/'latest').rename(path/'latest.previous')
            self.assertEqual(experiment.checkpoint_state(path,experiment.digest(config))['next_update'],1)

    def test_smoke_orchestration_resume_with_cpu_fixture_only(self):
        import torch
        from contextlib import nullcontext
        from transformers import LlamaConfig,LlamaForCausalLM
        class Tokenizer:
            def encode(self,text,**kwargs):
                return [int(text)] if text.isdigit() else [10]+([int(text[-1])] if text[-1].isdigit() else [])
        class Prepared(dict):
            def to(self,device):return self
        class Processor:
            tokenizer=Tokenizer()
            def apply_chat_template(self,messages,**kwargs):
                return 'fixture:' if kwargs['tokenize'] is False else Prepared(input_ids=torch.tensor([[10,11]]))
            def save_pretrained(self,path):experiment.atomic_json(Path(path)/'processor.json',{})
        def loader():
            return LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)),Processor(),[]
        scaler_type=torch.amp.GradScaler
        with tempfile.TemporaryDirectory() as tmp,patch.object(experiment.comparison.shared,'load_model',side_effect=loader),patch('torch.autocast',side_effect=lambda *a,**k:nullcontext()),patch('torch.amp.GradScaler',side_effect=lambda *a,**k:scaler_type('cpu',enabled=False)),redirect_stdout(io.StringIO()):
            root=Path(tmp)
            experiment.train(root,self.data,self.manifest,smoke=True)
            complete=experiment.read(root/'smoke-training/complete.json')
            self.assertEqual(complete['successful_updates'],1)
            self.assertEqual(complete['presentations'],4)
            losses=experiment.read(root/'smoke-training/losses.json')
            self.assertGreater(losses[0]['lora_b_matrices_changed'],0)
            weights=experiment.file_hash(root/'smoke-training/best_adapter/adapter_model.safetensors')
            experiment.train(root,self.data,self.manifest,resume=True,smoke=True)
            self.assertEqual(experiment.file_hash(root/'smoke-training/best_adapter/adapter_model.safetensors'),weights)
            self.assertEqual(len(experiment.read(root/'smoke-training/losses.json')),1)
            two_jobs=experiment.training_schedule(self.manifest)[:2]
            interrupted=root/'interrupted';reference=root/'reference'
            saver=experiment.save_checkpoint
            def interrupt_second(path,model,processor,state):
                if state['next_update']==2:raise OSError('simulated interruption before checkpoint commit')
                return saver(path,model,processor,state)
            with patch.object(experiment,'training_schedule',return_value=two_jobs):
                with patch.object(experiment,'save_checkpoint',side_effect=interrupt_second):
                    with self.assertRaisesRegex(OSError,'simulated'):
                        experiment.train(interrupted,self.data,self.manifest,smoke=True)
                experiment.train(interrupted,self.data,self.manifest,resume=True,smoke=True)
                experiment.train(reference,self.data,self.manifest,smoke=True)
            self.assertEqual(experiment.file_hash(interrupted/'smoke-training/best_adapter/adapter_model.safetensors'),
                             experiment.file_hash(reference/'smoke-training/best_adapter/adapter_model.safetensors'))
            self.assertEqual(experiment.read(interrupted/'smoke-training/losses.json'),
                             experiment.read(reference/'smoke-training/losses.json'))
            with self.assertRaises((ValueError,OSError)):experiment.require_selection(root,self.manifest)

    def assert_rng_equal(self,left,right):
        import numpy as np
        import torch
        self.assertEqual(left['python'],right['python'])
        self.assertEqual(left['numpy'][0],right['numpy'][0])
        self.assertTrue(np.array_equal(left['numpy'][1],right['numpy'][1]))
        self.assertEqual(left['numpy'][2:],right['numpy'][2:])
        self.assertTrue(torch.equal(left['torch'],right['torch']))
        self.assertEqual(len(left['cuda']),len(right['cuda']))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(left['cuda'],right['cuda'])))

    def retry_fixture(self,scale,always_overflow=False):
        import random
        import numpy as np
        import torch
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.lora_B=torch.nn.Parameter(torch.tensor([.2,-.3]))
        model=Model();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scaler=torch.amp.GradScaler('cpu',init_scale=scale)
        draws=[];clean=[]
        def loss(item):
            if item['set_id']=='a':clean.append(model.lora_B.grad is None)
            draw=(random.random(),float(np.random.random()),torch.rand(2))
            draws.append((draw[0],draw[1],draw[2].tolist()))
            return ((model.lora_B*draw[2]).sum()+draw[0]+draw[1]-item['target'])**2
        model.lora_B.register_hook(lambda g:torch.full_like(g,float('inf'))
            if always_overflow or scaler.get_scale()>=8 else g)
        return model,optimizer,scaler,loss,draws,clean

    def test_recover_skipped_amp_update_replays_same_rng_and_matches_clean_update(self):
        import torch
        batch=[dict(set_id='a',target=1.),dict(set_id='b',target=.5)]
        initial=experiment.capture_rng()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            model,optimizer,scaler,loss,draws,clean=self.retry_fixture(8)
            value,evidence=experiment.batch_update(model,optimizer,scaler,batch,loss,Path(tmp),1)
            after=experiment.capture_rng()
            self.assertEqual(evidence['retry_count'],1)
            self.assertEqual(evidence['attempted_scales'],[8.,4.])
            skipped=evidence['attempts'][0]
            self.assertFalse(skipped['optimizer_step_occurred'])
            self.assertEqual(skipped['nonfinite_gradient_values'],2)
            self.assertEqual(skipped['nonfinite_gradient_parameter_names'],['lora_B'])
            self.assertEqual(skipped['nonfinite_gradient_counts_by_parameter'],{'lora_B':2})
            self.assertTrue(evidence['optimizer_step_occurred']);self.assertTrue(evidence['finite_gradients'])
            self.assertGreater(evidence['lora_b_matrices_changed'],0)
            self.assertEqual(draws[:2],draws[2:]);self.assertEqual(clean,[True,True])
            self.assertEqual(int(optimizer.state[model.lora_B]['step']),1)
            self.assertFalse((Path(tmp)/'latest').exists())
            experiment.restore_rng(initial)
            reference,ref_opt,ref_scaler,ref_loss,ref_draws,_=self.retry_fixture(4)
            ref_value,_=experiment.batch_update(reference,ref_opt,ref_scaler,batch,ref_loss,Path(tmp),2)
            self.assertEqual(ref_value,value)
            self.assertTrue(torch.equal(model.lora_B,reference.lora_B))
            self.assertEqual(ref_draws,draws[:2])
            self.assert_rng_equal(after,experiment.capture_rng())

    def test_persistent_amp_failure_is_bounded_leaves_no_checkpoint_or_invalid_gradients(self):
        import torch
        batch=[dict(set_id='a',target=1.),dict(set_id='b',target=.5)]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            model,optimizer,scaler,loss,draws,clean=self.retry_fixture(8,True)
            before=model.lora_B.detach().clone();rng=experiment.capture_rng()
            with self.assertRaisesRegex(RuntimeError,'bounded AMP retries'):
                experiment.batch_update(model,optimizer,scaler,batch,loss,Path(tmp),7)
            report=experiment.read(Path(tmp)/'update_failure.json')
            self.assertEqual(report['planned_update'],7);self.assertEqual(report['retry_count'],4)
            self.assertEqual(report['attempted_scales'],[8.,4.,2.,1.,.5])
            self.assertEqual(len(report['attempts']),5)
            self.assertTrue(all(not r['optimizer_step_occurred'] for r in report['attempts']))
            self.assertTrue(all(r['nonfinite_gradient_values']==2 for r in report['attempts']))
            self.assertEqual(clean,[True]*5)
            self.assertTrue(all(draws[i:i+2]==draws[:2] for i in range(0,10,2)))
            self.assertIsNone(model.lora_B.grad);self.assertTrue(torch.equal(before,model.lora_B))
            self.assertFalse(optimizer.state);self.assertFalse((Path(tmp)/'latest').exists())
            self.assert_rng_equal(rng,experiment.capture_rng())
            self.assertEqual(scaler.get_scale(),.25)

    def test_finite_step_without_weight_change_is_not_retried(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.lora_B=torch.nn.Parameter(torch.zeros(1))
        model=Model();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scaler=torch.amp.GradScaler('cpu',init_scale=8)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'Unsafe/non-recoverable'):
                experiment.batch_update(model,optimizer,scaler,[dict(set_id='zero')],
                    lambda _:model.lora_B.sum()*0,Path(tmp),1)
            report=experiment.read(Path(tmp)/'update_failure.json')
            self.assertEqual(len(report['attempts']),1);self.assertTrue(report['attempts'][0]['optimizer_step_occurred'])
            self.assertEqual(report['attempts'][0]['lora_b_matrices_changed'],0)

    def test_matching_configs_rotations_metrics_and_resumption(self):
        jobs=experiment.evaluation_jobs(self.manifest,'validation')
        self.assertEqual(len(jobs),343)
        self.assertEqual(len(experiment.evaluation_jobs(self.manifest,'test')),686)
        before=experiment.configuration(self.manifest,'pretrained')
        after=experiment.configuration(self.manifest,'finetuned',adapter={'weight':'hash'})
        for field in ('model','training','manifest_hash','prompts','source_hashes','software'):
            self.assertEqual(before[field],after[field])
        rows=[]
        for job in jobs.values():
            label='7';pred=job['option_mapping'][label]
            probs={l:.4/6 for l in experiment.LABELS};probs[label]=.6
            rows.append(dict(job,probabilities=probs,predicted_label=label,predicted_rule=pred,
                             correct=pred==job['true_rule']))
        report=experiment.metrics(rows)
        self.assertAlmostEqual(report['accuracy'],1/7)
        self.assertEqual(report['sets'],49);self.assertEqual(report['rotation_consistency'],0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            for row in rows[:3]:experiment.append(path,row)
            self.assertEqual(len(experiment.load_rows(path,jobs)),3)
            experiment.append(path,rows[0])
            with self.assertRaises(ValueError):experiment.load_rows(path,jobs)
        for rule in experiment.comparison.IDS:
            group=[j for j in jobs.values() if j['true_rule']==rule and j['set_id']==next(s['set_id'] for s in self.manifest['sets'] if s['split']=='validation' and s['rule']==rule)]
            self.assertEqual({j['true_label'] for j in group},set(experiment.LABELS))
            self.assertEqual(len({tuple(j['image_files']) for j in group}),1)


if __name__=='__main__':unittest.main()
