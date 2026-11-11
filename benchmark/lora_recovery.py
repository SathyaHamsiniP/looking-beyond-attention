"""Read-only seven-rule LoRA artifact audit and trusted-checkpoint backup gate.

No GPU initialization, no automatic restart, and never a smoke adapter as final.
"""
import argparse
import json
import shutil
from pathlib import Path
import lora_7rules as seven


def audit(root):
    root=Path(root);training=root/'training'
    required=('config.json','losses.json','validation_losses.json','selection.json','complete.json',
        'latest/adapter_config.json','latest/adapter_model.safetensors','latest/training_state.pt',
        'best_adapter/adapter_config.json','best_adapter/adapter_model.safetensors','best_adapter/experiment.json')
    present={name:dict(bytes=(training/name).stat().st_size,sha256=seven.file_hash(training/name)) for name in required if (training/name).is_file()}
    losses=seven.read(training/'losses.json') if (training/'losses.json').is_file() else []
    report=dict(scope=str(root),present=present,missing=[name for name in required if name not in present],
        recorded_successful_updates=sum(bool(r.get('optimizer_step_occurred')) for r in losses),
        validation_loss_observations=len(seven.read(training/'validation_losses.json')) if (training/'validation_losses.json').is_file() else 0,
        unfinished_staging=bool((training/'latest.tmp').exists()),smoke_adapter_present=(root/'smoke-training/best_adapter').is_dir(),
        smoke_is_final_model=False,verified_full_run=False,verified_selected_adapter=False)
    complete_required=('config.json','complete.json','selection.json','validation_losses.json','best_adapter/adapter_config.json','best_adapter/adapter_model.safetensors','best_adapter/experiment.json')
    report['complete_candidates_present']=all(k in present for k in complete_required)
    report['resume_candidate_present']=all(k in present for k in ('config.json','latest/adapter_config.json','latest/adapter_model.safetensors','latest/training_state.pt'))
    if report['resume_candidate_present']:
        try:
            state=verify_checkpoint(training)
            report['verified_resume_update']=state['next_update']
        except Exception as exc:report['resume_error']=str(exc)
    report['decision']='Audit every available source before rerunning. Missing full artifacts are not a completed run. A staging directory is not resumable evidence.'
    print(json.dumps(report,indent=2),flush=True);return report


def verify_checkpoint(training):
    training=Path(training)
    if 'smoke' in str(training).lower():raise ValueError('Smoke model/checkpoint is never a full-run recovery source')
    config=seven.read(training/'config.json')
    if config.get('smoke_test') or config.get('kind')!='training' or config.get('model')!=seven.MODEL or config.get('training')!=seven.TRAIN:
        raise ValueError('Full seven-rule training configuration required')
    latest=training/'latest'
    for name in ('adapter_config.json','adapter_model.safetensors','training_state.pt'):
        if not (latest/name).is_file() or not (latest/name).stat().st_size:raise ValueError('Missing finalized latest checkpoint: '+name)
    import torch
    # Only the user's own trusted checkpoint may be supplied here.
    state=torch.load(latest/'training_state.pt',map_location='cpu',weights_only=False)
    n=state['next_update'];losses=state['losses']
    if state['config_hash']!=seven.digest(config) or not 1<=n<=111 or len(losses)!=n:
        raise ValueError('Checkpoint configuration/cursor/journal mismatch')
    if any(r['update']!=i+1 or not r['optimizer_step_occurred'] or not r['finite_gradients'] or r['lora_b_matrices_changed']<1 for i,r in enumerate(losses)):
        raise ValueError('Checkpoint lacks successful-update evidence')
    if not all(k in state for k in ('optimizer','scaler','rng','cuda_rng')):raise ValueError('Optimizer/scaler/RNG state missing')
    return state


def validate_backup(training):
    state=verify_checkpoint(training)
    print(f'VERIFIED trusted full-run checkpoint backup at successful update {state["next_update"]}; not a claim of completed training.',flush=True)
    return state


def snapshot_backup(training,destination):
    """Copy one stable finalized checkpoint; reject concurrent checkpoint swaps.

    A fresh local snapshot can then be uploaded with stable remote filenames.
    Hashes before/after copying prevent mixing optimizer/model updates.
    """
    training=Path(training);destination=Path(destination)
    if destination.exists():raise ValueError('Choose a fresh local backup snapshot directory')
    verify_checkpoint(training)
    names=('training_state.pt','adapter_model.safetensors','adapter_config.json')
    before={name:seven.file_hash(training/'latest'/name) for name in names}
    destination.mkdir(parents=True)
    try:
        shutil.copytree(training/'latest',destination/'latest')
        for name in ('config.json','answer_tokens.json','trainable.json'):
            if (training/name).is_file():shutil.copyfile(training/name,destination/name)
        after={name:seven.file_hash(training/'latest'/name) for name in names}
        copied={name:seven.file_hash(destination/'latest'/name) for name in names}
        if before!=after or before!=copied:raise ValueError('Checkpoint advanced while copying; retry a fresh snapshot')
        state=verify_checkpoint(destination)
        seven.atomic_json(destination/'losses.json',state['losses'])
        seven.atomic_json(destination/'validation_losses.json',state.get('validations',[]))
        if (training/'complete.json').exists():
            complete=seven.read(training/'complete.json')
            if complete['successful_updates']!=111 or complete['smoke_test'] or state['next_update']!=111:
                raise ValueError('Completion marker is not a verified full run')
            if complete['adapter_identity']!=seven.adapter_identity(training/'best_adapter'):
                raise ValueError('Selected adapter does not match completion identity')
            shutil.copytree(training/'best_adapter',destination/'best_adapter')
            if seven.adapter_identity(destination/'best_adapter')!=complete['adapter_identity']:raise ValueError('Selected adapter changed during snapshot')
            for name in ('complete.json','selection.json'):shutil.copyfile(training/name,destination/name)
        seven.atomic_json(destination/'backup_identity.json',dict(update=state['next_update'],checkpoint_sha256=copied,config_sha256=seven.file_hash(destination/'config.json')))
        print('Stable LoRA backup snapshot verified:',destination,flush=True)
        return destination
    except Exception:
        shutil.rmtree(destination)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--verify-checkpoint',action='store_true');p.add_argument('--backup-snapshot',type=Path)
    a=p.parse_args()
    if a.backup_snapshot:snapshot_backup(a.root/'training',a.backup_snapshot)
    elif a.verify_checkpoint:verify_checkpoint(a.root/'training');print('Finalized checkpoint verified; resume still requires exact runtime/source configuration checks.')
    else:audit(a.root)


if __name__=='__main__':main()
