"""CPU-only suite checks. No pretrained model initialization, inference or training."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import probing_7rules as probes
import import_results as imports
import lora_recovery
import iitm_drive
from probing_7rules import seven
import test_seven_rule_extensions as fixtures


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ExtensionTests.setUpClass();cls.fixture=fixtures.ExtensionTests
        cls.data=cls.fixture.data;cls.root=cls.fixture.root
        m=seven.builder.make_manifest(cls.data,seven.RULES_PATH,cls.root/'partitions.json',{'train':21,'validation':7},10,2026)
        old=seven.read(cls.fixture.manifest)
        m.update(rules=old['rules'],image_partitions=old['image_partitions'],partitions_canonical_hash=old['partitions_canonical_hash'])
        m['sets'].append(dict(split='test',image_files=['MUST_NOT_OPEN'],bits=['INVALID']))
        cls.manifest=cls.root/'saved-probe-manifest.json';seven.atomic_json(cls.manifest,m)
        cls.original=m;cls.items,cls.report=probes.plan_inputs(cls.manifest,cls.data)

    @classmethod
    def tearDownClass(cls):fixtures.ExtensionTests.tearDownClass()

    def test_all_five_subsets_identify_rule_and_original_manifest_is_unchanged(self):
        before=self.manifest.read_bytes();items,report=probes.plan_inputs(self.manifest,self.data)
        self.assertEqual(before,self.manifest.read_bytes());self.assertEqual(len(items),196)
        self.assertEqual(report['split_statistics']['train']['sets'],147)
        self.assertEqual(report['split_statistics']['validation']['sets'],49)
        self.assertEqual({v['minimum_distinct_configurations'] for v in report['minima'].values()},{5})
        for item in items:
            selected=report['selection'][item['set_id']]
            self.assertEqual(selected['10']['image_files'],item['image_files'])
            self.assertEqual(selected['5']['compatible_rules'],[item['rule']])
            self.assertEqual(len(set(selected['5']['bits'])),5)
            self.assertEqual(sorted(selected['5']['indices']),selected['5']['indices'])
            self.assertGreater(len(selected['1']['compatible_rules']),1)

    def test_one_image_is_label_blind_and_ambiguous(self):
        names=self.items[0]['image_files'];one=names[probes.select_one(names)]
        self.assertEqual(one,list(reversed(names))[probes.select_one(list(reversed(names)))])
        for rule in self.report['rules']['rules']:
            for bit in rule['true_bits']:
                self.assertGreater(len(seven.builder.compatible_rules([bit],self.report['rules']['rules'])),1)

    def test_test_images_not_opened_and_train_validation_no_leakage(self):
        original=seven.file_hash;assign=self.original['image_partitions']
        def guard(path):
            p=Path(path)
            if p.suffix=='.png' and assign[p.name]=='test':raise AssertionError('Held-out test image opened')
            return original(path)
        with patch.object(seven,'file_hash',side_effect=guard):probes.plan_inputs(self.manifest,self.data)
        bad=copy.deepcopy(self.original);bad['sets'][0]['image_files'][0]=next(n for n,v in assign.items() if v=='validation')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json';seven.atomic_json(path,bad)
            with self.assertRaises(ValueError):probes.plan_inputs(path,self.data)

    def test_normalization_fits_only_training_and_shuffle_does_not_mutate_labels(self):
        x=np.arange(14,dtype=float).reshape(-1,1);y=np.array([f'R{i%7+1}' for i in range(14)]);before=y.copy()
        model,_,_=probes.fit_layer(x,y,np.full((7,1),10000.))
        self.assertTrue(np.allclose(model.named_steps['standardscaler'].mean_,x.mean(0)))
        self.assertEqual(model.named_steps['standardscaler'].n_samples_seen_,14)
        a,pa,_=probes.fit_layer(x,y,np.arange(7).reshape(-1,1),shuffled=True)
        b,pb,_=probes.fit_layer(x,y,np.arange(7).reshape(-1,1),shuffled=True)
        self.assertTrue(np.array_equal(pa,pb));self.assertTrue(np.array_equal(y,before))

    def test_feature_extraction_resume_recovers_orphan_metadata_without_model(self):
        helpers=probes.capture_helpers()
        class Tokenizer:
            def encode(self,text,**kwargs):return [int(text)]
        def capture(*args):return np.ones((32,7),dtype=np.float32),{l:1/7 for l in seven.LABELS},10
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()),patch.object(probes.importlib.metadata,'version',side_effect=lambda name:probes.INFERENCE_PACKAGES[name]):
            out=Path(tmp)/'features'
            with patch.object(probes,'capture_helpers',return_value=helpers),patch.object(helpers,'capture',side_effect=capture),patch.object(probes.choices.shared,'load_model',return_value=(None,SimpleNamespace(tokenizer=Tokenizer()),[])) as load:
                probes.extract(out,self.data,self.items,self.report,0,smoke=True);load.assert_called_once()
            (out/'model_predictions.jsonl').unlink()
            with patch.object(probes.choices.shared,'load_model',side_effect=AssertionError('Resumed complete extraction loaded model')):
                probes.extract(out,self.data,self.items,self.report,0,resume=True,smoke=True)
            self.assertEqual(len((out/'model_predictions.jsonl').read_text().splitlines()),3)
            row=(out/'model_predictions.jsonl').read_text().splitlines()[0]
            with (out/'model_predictions.jsonl').open('a') as f:f.write(row+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                probes.extract(out,self.data,self.items,self.report,0,resume=True,smoke=True)

    def test_invalid_feature_checksum_missing_file_and_shapes_rejected(self):
        helpers=probes.capture_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);helpers.save_features(folder/'features.npz',np.ones((31,2)),10)
            expected={'set_id':'a'};row=dict(expected,input_tokens=10,probabilities={l:1/7 for l in seven.LABELS},feature_sha256=seven.file_hash(folder/'features.npz'))
            seven.atomic_json(folder/'metadata.json',row)
            with self.assertRaisesRegex(ValueError,'32'):probes.feature_record(folder,expected,helpers)
            (folder/'features.npz').unlink()
            with self.assertRaises(OSError):probes.feature_record(folder,expected,helpers)

    def test_safe_results_import_model_separation_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();seven.atomic_json(source/'config.json',dict(model='Qwen/Qwen3.5-4B'))
            (source/'summary.json').write_text('{"measured":true}')
            (source/'activation.npz').write_bytes(b'not imported');original=(source/'summary.json').read_bytes()
            dest=Path(tmp)/'seven-rules/qwen3.5-4b/probing'
            imports.import_folder(source,dest,7,'probing')
            self.assertEqual(original,(source/'summary.json').read_bytes());self.assertFalse((dest/'activation.npz').exists())
            self.assertEqual(seven.file_hash(dest/'summary.json'),seven.file_hash(source/'summary.json'))
            seven.atomic_json(source/'config.json',dict(model='Qwen/Qwen3-VL-4B'))
            with self.assertRaises(ValueError):imports.import_folder(source,dest,7,'probing')
            with self.assertRaises(ValueError):imports.import_folder(source,source/'seven-rules/qwen3.5-4b/probing',7,'probing')

    def test_partial_checkpoint_and_smoke_are_never_final_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training';training.mkdir()
            seven.atomic_json(training/'config.json',dict(kind='training',smoke_test=False,model=seven.MODEL,training=seven.TRAIN))
            (training/'latest.tmp').mkdir();(training/'latest.tmp/adapter_model.safetensors').write_bytes(b'partial')
            report=lora_recovery.audit(root)
            self.assertFalse(report['resume_candidate_present']);self.assertTrue(report['unfinished_staging']);self.assertFalse(report['verified_full_run'])
            with self.assertRaises(ValueError):lora_recovery.verify_checkpoint(training)
            with self.assertRaises(ValueError):lora_recovery.verify_checkpoint(root/'smoke-training')

    def test_notebook_authentication_never_calls_colab_kernel_in_subprocess(self):
        with patch('IPython.get_ipython',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'not a subprocess'):iitm_drive.connect(authenticate=True)
        google=ModuleType('google');auth=ModuleType('google.auth');colab=ModuleType('google.colab');colab_auth=ModuleType('google.colab.auth')
        from unittest.mock import Mock
        colab_auth.authenticate_user=Mock();colab.auth=colab_auth;auth.default=Mock(return_value=(object(),None));google.auth=auth
        api=ModuleType('googleapiclient');discovery=ModuleType('googleapiclient.discovery');discovery.build=Mock(return_value=object());api.discovery=discovery
        modules={'google':google,'google.auth':auth,'google.colab':colab,'google.colab.auth':colab_auth,'googleapiclient':api,'googleapiclient.discovery':discovery}
        with patch.dict(sys.modules,modules),patch('IPython.get_ipython',return_value=SimpleNamespace(kernel=object())),patch.object(iitm_drive,'validate_destination') as verify:
            service=iitm_drive.connect(authenticate=True);colab_auth.authenticate_user.assert_called_once();verify.assert_called_once_with(service)
        colab_auth.authenticate_user.reset_mock()
        with patch.dict(sys.modules,modules),patch('IPython.get_ipython',return_value=None),patch.object(iitm_drive,'validate_destination'):
            iitm_drive.connect();colab_auth.authenticate_user.assert_not_called()

    def test_stable_checkpoint_snapshot_and_corrupt_cursor_rejection(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training';latest=training/'latest';latest.mkdir(parents=True)
            config=dict(kind='training',smoke_test=False,model=seven.MODEL,training=seven.TRAIN)
            seven.atomic_json(training/'config.json',config)
            seven.atomic_json(latest/'adapter_config.json',{'r':8})
            (latest/'adapter_model.safetensors').write_bytes(b'CPU fixture, not an actual adapter')
            loss=dict(update=1,optimizer_step_occurred=True,finite_gradients=True,lora_b_matrices_changed=248)
            state=dict(config_hash=seven.digest(config),next_update=1,losses=[loss],validations=[],optimizer={},scaler={},rng=torch.get_rng_state(),cuda_rng=[])
            torch.save(state,latest/'training_state.pt')
            snapshot=root/'backup';lora_recovery.snapshot_backup(training,snapshot)
            self.assertEqual(lora_recovery.verify_checkpoint(snapshot)['next_update'],1)
            self.assertEqual(seven.read(snapshot/'losses.json'),[loss])
            with self.assertRaises(ValueError):lora_recovery.snapshot_backup(training,snapshot)
            state['next_update']=2;torch.save(state,latest/'training_state.pt')
            with self.assertRaises(ValueError):lora_recovery.snapshot_backup(training,root/'corrupt')
            self.assertFalse((root/'corrupt').exists())

    def test_pinned_inference_packages_reject_mismatch_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(probes.importlib.metadata,'version',return_value='wrong'),patch.object(probes.choices.shared,'load_model') as load,redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'pinned seven-rule'):
                probes.extract(Path(tmp)/'out',self.data,self.items,self.report,0,smoke=True)
            load.assert_not_called()


if __name__=='__main__':unittest.main()
