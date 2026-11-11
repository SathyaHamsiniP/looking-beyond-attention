# 06 — Seven-rule LoRA

Does low-rank adaptation improve rule identification beyond the pretrained model? The intended design uses the same saved seven-rule dataset, 147 training sets and 49 validation sets; test is reserved. Pretrained validation accuracy is **48/343** across seven rotations. A selected full-run fine-tuned evaluation is **unavailable**.

## Results

The [saved file inventory](../../results/seven-rules/qwen3.5-4b/lora/artifact_audit.json) lists files in both the initial and retry folders, including their subfolders and the available notebook. The retry has the original dataset, 343 pretrained predictions, full-training settings and **one successful optimizer-update record**, empty validation losses and an empty `latest.tmp`. It has no finalized full-run `latest`, completion/selection record, selected adapter or fine-tuned predictions. The completed one-update smoke adapter and optimizer binaries exist, but are not a final trained model. No adapter/optimizer binaries were imported into Git.

The notebook claim **116/343 after LoRA is unverified** without the underlying evaluation. The available files do not support that score. Search any remaining local backups before rerunning. If no additional finalized checkpoint is recovered, a fresh full run is required; no new training was run here. Never resume `latest.tmp` or silently promote smoke weights.

## Keeping the completed results

Once a completed run is verified, keep **only compact evidence** in [the LoRA results folder](../../results/seven-rules/qwen3.5-4b/lora/README.md): the actual full-run training configuration and losses, epoch-level validation losses, completion and checkpoint-selection records, fine-tuned validation predictions and metrics, and a matched pretrained-versus-fine-tuned comparison. Include existing plots only if they help interpret the result. Preserve the original filenames and recorded model/checkpoint identities.

The full training archive, adapter weights (`.safetensors`), optimizer/scaler/RNG states, intermediate checkpoints, smoke runs and duplicate datasets stay in a separate Drive backup, **not GitHub**. The pretrained baseline and 343 detailed predictions already exist under [operator controls](../../results/seven-rules/qwen3.5-4b/operator-controls/README.md); link to them rather than uploading a second copy.

After the completed run has been checked, replace the obsolete `partial-diagnostics/` and `artifact_audit.json` in the **active results folder**, retaining historical provenance in Git history. Do not describe the run as complete or report a fine-tuned accuracy until its selected adapter, completion record and saved prediction counts have been independently reconciled. Do not rerun experiments merely for documentation.

## Method

Qwen3.5-4B uses NF4 four-bit base weights with double quantization and FP16 computation. LoRA rank 8, alpha 16, dropout 0.05 and the existing linear target-module resolver remain unchanged. AdamW uses learning rate 5e-5, betas (0.9,0.999), epsilon 1e-8 and weight decay 0.01. Initial gradient scale is 8. Three epochs, accumulation four sets, 441 presentations and **111 successful optimizer updates** are planned. Checkpoint selection minimizes full-vocabulary answer-token validation loss at epoch ends, not accuracy. The existing bounded overflow retry, RNG restoration and successful-update checks are retained in [the runner](../../code/06-lora/lora_7rules.py).

## How to run

```python
!git clone --branch feature/organize-benchmark https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
%cd /content/looking-beyond-attention
!pip install -r benchmark/code/00-common/requirements-seven-rule-controls.txt
from google.colab import drive
from pathlib import Path
import subprocess, sys
drive.mount('/content/iitm-drive')  # Choose the School account.
SOURCE = Path('/content/iitm-drive/MyDrive/LogicCBM++/source-research')
DATA = SOURCE / 'three-operand-experiments/data-1000'
OLD = SOURCE / 'benchmark-pilot/lora-7rules-fp16-retry'
MANIFEST = OLD / 'dataset/manifest.json'
```

The source is a read-only shared input. If the saved manifest or images are absent, recover the exact files rather than building replacement sets. The pinned model is `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, NF4 4-bit with double quantization, FP16 computation, SDPA and max pixels 100352. Match the recorded inference environment, including PyTorch `2.14.1+cu130`, before GPU execution; install its matching CUDA build and restart if necessary. Package checks reject mismatches.

### Continue a saved run or start a new one

```python
subprocess.run(['python','benchmark/code/tools/lora_recovery.py','--root',str(OLD)],check=True)
```

Only copy a trusted finalized backup locally after checking its entire source directory. `training/config.json`, `latest` model/optimizer/scaler/RNG state and matching journals must belong to the same full run. Then:

```python
# LOCAL is a copied backup root, never the original shared folder.
LOCAL = Path('/content/seven-rule-lora-recovered')
subprocess.run(['python','benchmark/code/tools/lora_recovery.py','--root',str(LOCAL),
                '--verify-checkpoint'],check=True)
# Use the checkpoint's original commit; shown here for pre-move backups.
ORIGINAL = Path('/content/7-rule-original')
if not ORIGINAL.exists():
    subprocess.run(['git','worktree','add','--detach',str(ORIGINAL),
                    '6d13d992432a9864b2879074f0b2e9d750bbdfa9'],check=True)
subprocess.run(['python',str(ORIGINAL/'benchmark/lora_7rules.py'),'train','--data-dir',str(DATA),
                '--out',str(LOCAL),'--resume'],check=True)
```

The original runner enforces configuration, source files, data order and package versions on resume. If verification fails, stop and investigate; do not restart into that folder. A partial journal without finalized weights is not resumable.

If no complete checkpoint is found in any backup, prepare a **fresh**, separate output root. Copy the saved dataset directory byte-for-byte; do not call `build` or modify the original manifest. A fresh run with the moved code requires a baseline from that same code and packages. Do not copy the old baseline config and replace its hashes. To reuse the existing baseline without new inference, run both baseline checks and training from its matching original checkout instead. See [earlier-run notes](../../code/README.md#earlier-saved-runs).

```python
import shutil
LOCAL = Path('/content/seven-rule-lora-new')
assert not LOCAL.exists()
LOCAL.mkdir()
shutil.copytree(OLD/'dataset', LOCAL/'dataset')
base = ['python','benchmark/code/06-lora/lora_7rules.py','--data-dir',str(DATA),'--out',str(LOCAL)]
subprocess.run(base + ['train','--validate-only'],check=True) # CPU checks and planned update count only
# GPU commands, run manually only after checking all backups:
# The moved code requires a baseline with its own exact source/package hashes.
# subprocess.run(base + ['baseline'],check=True) # 343 validation forwards
# subprocess.run(base + ['train'],check=True)
# subprocess.run(base + ['evaluate'],check=True) # Selected adapter, validation only
# subprocess.run(base + ['compare'],check=True)
```

No final-test command is part of this procedure. Successful updates checkpoint locally before mirrored diagnostics are written. Evaluation checks that training is complete and the selected adapter matches its saved record. Console output reports updates and validation findings; preserve all diagnostics even if upload fails.

### Back up to School Drive

Create snapshots between updates, preferably while training is paused. The helper hashes before and after copying and rejects any concurrent checkpoint swap; the saved optimizer step determines which loss records belong in the backup. Use a fresh snapshot path each time; remote uploads reuse stable filenames in experiment 06.

```python
sys.path.insert(0,'/content/looking-beyond-attention/benchmark/code/tools')
import lora_recovery, iitm_drive
SNAPSHOT = Path('/content/lora-backup-update-001') # Fresh path for each snapshot.
lora_recovery.snapshot_backup(LOCAL/'training', SNAPSHOT)
service = iitm_drive.connect(authenticate=True) # Notebook cell; select School account.
iitm_drive.upload(service, SNAPSHOT, 'lora', include_model_state=True)
# Also back up dataset manifest, evaluation predictions and comparison outputs:
iitm_drive.upload(service, LOCAL, 'lora') # Compact files; weights excluded by default.
```

The uploader checks `24f2005474@ds.study.iitm.ac.in`, folder `15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR`, file hashes and sizes. A failed upload can leave only part of a checkpoint on Drive: keep the local snapshot until every receipt succeeds, and verify a downloaded backup before resume. If quota fails, retain `/content` and download a ZIP made with `iitm_drive.manual_zip(SNAPSHOT, Path('/content/lora-backup.zip'), include_model_state=True)`. Never fall back to personal Drive. Backups must include weights, optimizer/scaler/RNG state, config, validation losses, selection and selected adapter when complete; these stay outside Git.

## Limitations

Three-rule LoRA used two epochs and 46 updates, so it is not matched to this 111-update design. Seven rotations reuse the same image set and are not independent trials; images also recur within partitions. A recovered validation improvement alone would not show Boolean rule learning. [Current compact evidence](../../results/seven-rules/qwen3.5-4b/lora/README.md).
