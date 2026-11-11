"""CPU tests for seven-rule probing, result imports and backups. No pretrained model initialization, inference or training."""
import setup_paths
from benchmark_paths import BENCHMARK, source_path
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
import test_controls_patching as fixtures


class ExperimentTests(unittest.TestCase):
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





    def test_pinned_inference_packages_reject_mismatch_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(probes.importlib.metadata,'version',return_value='wrong'),patch.object(probes.choices.shared,'load_model') as load,redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'pinned seven-rule'):
                probes.extract(Path(tmp)/'out',self.data,self.items,self.report,0,smoke=True)
            load.assert_not_called()


if __name__=='__main__':unittest.main()
