"""CPU-only tests for validation recovery, historical hooks and IITM upload guards."""
from collections import Counter
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import seven_rule_validation as validation
import activation_patching_7rules as patching
import operator_controls_7rules as controls
import iitm_drive as drive
from seven_rule_validation import seven,choices


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.data=cls.root/'data';(cls.data/'images').mkdir(parents=True)
        records=[];assign={}
        for n in range(8):
            bits=list(map(int,f'{n:03b}'))
            for i in range(125):
                name=f'scene_{n:03b}_{i:03d}.png';(cls.data/'images'/name).write_bytes(b'fixture')
                records.append(dict(filename=name,bits=bits,split='train' if i<100 else 'test'))
                assign[name]='train' if i<100 else 'validation' if i<112+n%2 else 'test'
        seven.atomic_json(cls.data/'metadata.json',dict(records=records));part=cls.root/'partitions.json';seven.atomic_json(part,assign)
        m=validation.builder.make_manifest(cls.data,seven.RULES_PATH,part,{'validation':7},10,2026)
        m.update(protocol='seven-generating-rules-qlora-v1',rules=choices.validate_rules(seven.RULES_PATH),image_partitions=assign,partitions_canonical_hash=seven.digest(assign))
        # A forbidden non-validation input is deliberately not a valid set.
        m['sets'].append(dict(split='test',image_files=['DO_NOT_OPEN.png'],bits=['INVALID']))
        cls.manifest=cls.root/'manifest.json';seven.atomic_json(cls.manifest,m)
        cls.view=validation.validation_view(cls.manifest,cls.data)
        cls.jobs=validation.jobs(cls.view)
        cls.baseline=cls.root/'baseline';cls.baseline.mkdir()
        cfg=seven.configuration(m,'pretrained');cfg['source_hashes']['lora_7rules.py']=sorted(validation.TRUSTED_INPUT_HELPERS)[0]
        seven.atomic_json(cls.baseline/'config.json',cfg)
        rows={}
        for key,job in cls.jobs.items():
            probs={l:.4/6 for l in seven.LABELS};probs['7']=.6;pred=job['option_mapping']['7']
            rows[key]=dict(job,probabilities=probs,predicted_label='7',predicted_rule=pred,correct=pred==job['true_rule'])
        cls.rows=rows
        (cls.baseline/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows.values()))
        seven.atomic_json(cls.baseline/'metrics.json',seven.metrics(list(rows.values())))
        seven.atomic_json(cls.baseline/'answer_tokens.json',dict(zip(seven.LABELS,range(7))))

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_validation_only_no_nonvalidation_image_access(self):
        original=seven.file_hash
        selected=set(self.view['validation_image_sha256'])
        def guarded(path):
            p=Path(path)
            if p.suffix=='.png' and p.name not in selected:raise AssertionError('Non-validation image access')
            return original(path)
        with patch.object(seven,'file_hash',side_effect=guarded):
            view=validation.validation_view(self.manifest,self.data)
        self.assertEqual(len(view['sets']),49)
        self.assertEqual(Counter(s['rule'] for s in view['sets']),{r:7 for r in choices.IDS})
        self.assertEqual(len(validation.jobs(view)),343)

    def test_exact_recovery_rejects_metadata_model_prompt_and_duplicate_corruption(self):
        rows,_,_=validation.verify_baseline(self.baseline,self.view)
        self.assertEqual(len(rows),343)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            import shutil
            for p in self.baseline.iterdir():shutil.copyfile(p,out/p.name)
            cfg=seven.read(out/'config.json');cfg['model']['revision']='wrong';seven.atomic_json(out/'config.json',cfg)
            with self.assertRaisesRegex(ValueError,'identity'):validation.verify_baseline(out,self.view)
            shutil.copyfile(self.baseline/'config.json',out/'config.json')
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(next(iter(rows.values())))+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):validation.verify_baseline(out,self.view)

    def test_pair_balance_controls_and_shared_input_membership(self):
        pairs=patching.make_pairs(self.view['sets'])
        self.assertEqual(len(pairs),14);self.assertEqual(len({p['pair_id'] for p in pairs}),14)
        for condition in ('same','different'):
            self.assertEqual(Counter(p['receiver_rule'] for p in pairs if p['condition']==condition),{r:1 for r in choices.IDS})
            self.assertEqual(Counter(p['donor_rule'] for p in pairs if p['condition']==condition),{r:1 for r in choices.IDS})
        for p in pairs:
            self.assertNotEqual(p['receiver_set'],p['donor_set'])
            self.assertEqual(p['receiver_rule']==p['donor_rule'],p['condition']=='same')
        with self.assertRaises(ValueError):patching.make_pairs(self.view['sets'],8)

    def test_true_rule_answer_and_perturbation_are_distinct_signed_effects(self):
        pair=patching.make_pairs(self.view['sets'])[1]
        before={r:.1 for r in choices.IDS};before['R7']=.4
        donor={r:.1 for r in choices.IDS};donor['R7']=.4
        after=before.copy();after[pair['donor_rule']]-=.05;after['R7']+=.05
        row=patching.effect(pair,0,1,'final',before,donor,after)
        self.assertAlmostEqual(row['delta_true_rule'],-.05)
        self.assertAlmostEqual(row['delta_donor_answer'],.05)
        self.assertAlmostEqual(row['total_variation'],.05)
        self.assertNotEqual(row['donor_true_rule'],row['donor_predicted_rule'])
        self.assertEqual(row['receiver_original_predicted_rule'],'R7')

    def test_set_bootstrap_and_operator_analysis_no_inference(self):
        report=controls.analyze(self.rows,self.view)
        self.assertAlmostEqual(report['accuracy'],1/7)
        self.assertEqual(report['rotation_consistency'],0)
        self.assertAlmostEqual(report['accuracy_95ci_set_cluster'][0],1/7)
        self.assertAlmostEqual(report['accuracy_95ci_set_cluster'][1],1/7)
        self.assertEqual(validation.bootstrap([1.,2.]),validation.bootstrap([1.,2.]))

    def test_pair_cluster_aggregation_and_controls(self):
        pairs=patching.make_pairs(self.view['sets']);rows=[]
        before={r:1/7 for r in choices.IDS}
        for pair in pairs:
            after=before.copy();after[pair['donor_rule']]+=.01
            other=next(r for r in choices.IDS if r!=pair['donor_rule']);after[other]-=.01
            for rotation in (0,1):rows.append(patching.effect(pair,rotation,1,'final',before,before,after))
        report=patching.summarize(rows)
        self.assertTrue(all(r['pair_clusters']==7 for r in report['layer_effects']))
        self.assertAlmostEqual(report['matched_controls'][0]['different_minus_same'],0)

    def test_historical_hooks_shape_positions_and_cleanup(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()]))
                self.fail=False
            def forward(self,input_ids,**kwargs):
                h=input_ids
                for layer in self.model.layers:h=layer(h)
                self.last=h.detach().clone()
                if self.fail:raise RuntimeError('injected failure')
                return SimpleNamespace(logits=h)
        engine=patching.intervention_engine(list(range(7)));model=Model();wrapper=SimpleNamespace(model=model)
        mapping=choices.mapping(7,0);donor=torch.arange(21,dtype=torch.float32).reshape(1,3,7)
        receiver=torch.zeros_like(donor)
        capture=engine.record_original(wrapper,{'input_ids':donor},mapping,model.model.layers)
        self.assertEqual(set(capture['activations']),{1,2})
        self.assertEqual(capture['activations'][1]['final'].shape,(7,))
        engine.patched_probabilities(wrapper,{'input_ids':receiver},mapping,model.model.layers[0],capture['activations'][1]['previous'],'previous')
        self.assertTrue(torch.equal(model.last[0,-2],donor[0,-2]));self.assertTrue(torch.equal(model.last[0,-1],receiver[0,-1]))
        self.assertTrue(all(not b._forward_hooks for b in model.model.layers))
        model.fail=True
        with self.assertRaises(RuntimeError):engine.record_original(wrapper,{'input_ids':donor},mapping,model.model.layers)
        self.assertTrue(all(not b._forward_hooks for b in model.model.layers))

    def test_local_output_guards_and_resume(self):
        with self.assertRaises(ValueError):validation.local_path('/content/source-drive/MyDrive/new-results')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result';cfg={'dataset':'validation'}
            with seven.output(p,cfg):pass
            with self.assertRaises(ValueError):
                with seven.output(p,{'dataset':'different'},True):pass
            with seven.output(p,cfg,True):pass

    def test_upload_destination_identity_and_permission_fail_closed(self):
        class Service:
            def __init__(self,email,writable=True):self.email=email;self.writable=writable
            def about(self):return self
            def files(self):return self
            def get(self,**kw):
                row={'user':{'emailAddress':self.email}} if 'fileId' not in kw else dict(id=drive.FOLDER,name='School',mimeType='application/vnd.google-apps.folder',capabilities={'canAddChildren':self.writable})
                return SimpleNamespace(execute=lambda:row)
        self.assertEqual(drive.EMAIL,'24f2005474@ds.study.iitm.ac.in')
        # Regression: the former, incorrect domain must fail before any upload work.
        with patch.object(drive,'folder_child',side_effect=AssertionError('Wrong account reached a write')):
            with self.assertRaisesRegex(ValueError,'No upload attempted'):
                drive.upload(Service('24f2005474@ds.iitm.study.in'),Path('/nonexistent-results'),'operator-controls')
        with self.assertRaisesRegex(ValueError,'No upload attempted'):drive.validate_destination(Service('sathyahamsinip@gmail.com'))
        with self.assertRaisesRegex(ValueError,'writable'):drive.validate_destination(Service(drive.EMAIL,False))
        self.assertEqual(drive.validate_destination(Service(drive.EMAIL))['id'],drive.FOLDER)
        self.assertEqual(drive.FOLDER,'15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR')

    def test_resumable_upload_reuses_file_id_and_skips_identical_bytes(self):
        import hashlib,sys,types
        class Service:
            def __init__(self):
                self.nodes={drive.FOLDER:dict(id=drive.FOLDER,name='School',mimeType='application/vnd.google-apps.folder',capabilities={'canAddChildren':True})};self.updates=0
            def files(self):return self
            def about(self):return self
            def get(self,**kw):return SimpleNamespace(execute=lambda:self.nodes[kw['fileId']] if 'fileId' in kw else {'user':{'emailAddress':drive.EMAIL}})
            def create(self,body,**kw):
                identity=str(len(self.nodes));self.nodes[identity]=dict(body,id=identity)
                return SimpleNamespace(execute=lambda:self.nodes[identity])
            def update(self,fileId,media_body,**kw):
                self.updates+=1
                def chunk(**kwargs):
                    data=Path(media_body.path).read_bytes();self.nodes[fileId].update(size=str(len(data)),md5Checksum=hashlib.md5(data).hexdigest())
                    return None,self.nodes[fileId]
                return SimpleNamespace(next_chunk=chunk)
        service=Service()
        def matches(s,parent,name):return [v for v in s.nodes.values() if v.get('name')==name and parent in v.get('parents',[])]
        package=types.ModuleType('googleapiclient');http=types.ModuleType('googleapiclient.http')
        http.MediaFileUpload=lambda path,**kw:SimpleNamespace(path=path);package.http=http
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'googleapiclient':package,'googleapiclient.http':http}),patch.object(drive,'matches',side_effect=matches):
            source=Path(tmp);(source/'summary.json').write_text('{"accuracy":0.14}')
            first=drive.upload(service,source,'operator-controls');second=drive.upload(service,source,'operator-controls')
            self.assertEqual(service.updates,1)
            self.assertEqual(first['files'][0]['file_id'],second['files'][0]['file_id'])
            self.assertEqual(len(service.nodes),3)

    def test_completed_patching_resume_has_zero_model_load_and_rejects_duplicates(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache';cache.mkdir()
            for source in self.baseline.iterdir():shutil.copyfile(source,cache/('recovered_config.json' if source.name=='config.json' else source.name))
            seven.atomic_json(cache/'validation_view.json',self.view)
            out=Path(tmp)/'patches';pairs=patching.make_pairs(self.view['sets'])[:2]
            with patch.object(choices.shared,'load_model',side_effect=RuntimeError('fixture stops before model initialization')):
                with self.assertRaisesRegex(RuntimeError,'fixture stops'):
                    patching.run(self.view,self.data,cache,out,pairs,[0],[1],['final'])
            probabilities={r:1/7 for r in choices.IDS}
            rows=[dict(patching.effect(pair,0,1,'final',probabilities,probabilities,probabilities),key=f"{pair['pair_id']}:0:1:final") for pair in pairs]
            for row in rows:seven.append(out/'patches.jsonl',row)
            with patch.object(choices.shared,'load_model',side_effect=AssertionError('Completed resume loaded model')):
                patching.run(self.view,self.data,cache,out,pairs,[0],[1],['final'],True)
            seven.append(out/'patches.jsonl',rows[0])
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                patching.run(self.view,self.data,cache,out,pairs,[0],[1],['final'],True)

    def test_manual_zip_is_compact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'results';source.mkdir();(source/'summary.json').write_text('{}')
            (source/'adapter.safetensors').write_bytes(b'not included');(source/'.credentials.json').write_text('never included')
            target=Path(tmp)/'fallback.zip';drive.manual_zip(source,target)
            import zipfile
            with zipfile.ZipFile(target) as archive:self.assertEqual(archive.namelist(),['summary.json'])
            with self.assertRaises(ValueError):drive.manual_zip(source,target)


if __name__=='__main__':unittest.main()
