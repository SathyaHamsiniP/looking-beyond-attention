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
import iitm_drive as drive



class ToolTests(unittest.TestCase):
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


    def test_manual_zip_is_compact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'results';source.mkdir();(source/'summary.json').write_text('{}')
            (source/'adapter.safetensors').write_bytes(b'not included');(source/'.credentials.json').write_text('never included')
            target=Path(tmp)/'fallback.zip';drive.manual_zip(source,target)
            import zipfile
            with zipfile.ZipFile(target) as archive:self.assertEqual(archive.namelist(),['summary.json'])
            with self.assertRaises(ValueError):drive.manual_zip(source,target)


if __name__=='__main__':unittest.main()
