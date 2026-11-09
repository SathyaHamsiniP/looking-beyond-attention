# 07 — Seven-rule operator controls (A3)

Does pretrained Qwen3.5-4B identify seven generating Boolean rules from ten positive images? This extends the existing A1/A2 comparison descriptively; it does not repeat A1 (three rules/three choices) or A2 (the same three rules/seven choices). A3 uses different validation sets, so differences do not isolate the effect of generating-rule count.

Rules are loaded unchanged from [`rules_7.json`](../../rules_7.json): R1 `(A AND B) OR C`, R2 `(A AND C) OR B`, R3 `(B AND C) OR A`, R4 `(A XOR B) OR C`, R5 `(A XOR C) OR B`, R6 `(B XOR C) OR A`, R7 `(A OR B OR C) AND NOT(A AND B AND C)`. A/B/C are green cube/blue sphere/red cylinder, requiring exact colour and shape; other objects are distractors.

## Recovered results and scope

The saved `benchmark-pilot/lora-7rules-fp16-retry` manifest and pretrained baseline were recovered from the original Drive. All **343 per-example predictions** matched the 49 validation sets, seven rules, seven rotations, pinned model/settings, canonical prompts and audited inference-source hashes. Recomputed metrics match the saved metrics:

- **48/343 = 13.9942%**, versus chance 14.2857%; chance-adjusted accuracy −0.00340.
- Positions 1–5: zero; position 6: 82; position 7: 261.
- Rule consistency: zero of 49 sets.
- Set-clustered 95% bootstrap interval: approximately 11.9534–16.0350% (5,000 draws, seed 2026).

These numbers were recovered and analyzed on CPU, with **zero new model forwards**. Current validation-image hashes are recorded to bind later cache use; the original baseline stored a full-dataset fingerprint rather than individual image hashes. Reuse assumes the original read-only source images have not been altered; the tool checks metadata, full manifest identity, filenames/configurations, validation-file availability and then records current validation-image hashes. No held-out test examples are selected, prepared or scored. Partition tags and opaque whole-manifest hashes identify the supplied file.

The saved A1/A2 summary reports 36.67% (three choices) and 14.76% (seven choices), both on the original three generating rules. Only their recorded aggregates are used; no new old-task inference, reconstructed predictions, cross-dataset matched comparisons or old-task confidence intervals are fabricated.

## Colab: fresh runtime and exact paths

```python
!git clone --branch feature/seven-rule-controls-patching https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
%cd /content/looking-beyond-attention
!pip install -r benchmark/requirements-seven-rule-controls.txt
```

For CPU A3 analysis, CUDA/PyTorch is not needed. For later patching, the recovered baseline used **PyTorch 2.14.1+cu130**; check `import torch; print(torch.__version__)` before using GPU quota. The patch runner refuses mismatched inference packages. If needed, install the matching official PyTorch build and restart the runtime; do not silently compare different inference environments.

Mount the original account only to **read** its existing files. Explicitly choose `sathyahamsinip@gmail.com` in the mount dialog. This is separate from the IITM Drive API authentication used for all new uploads.

```python
from google.colab import drive
from pathlib import Path
import subprocess

drive.mount('/content/source-drive')
SOURCE = Path('/content/source-drive/MyDrive/LogicCBM++/looking-beyond-attention')
DATA = SOURCE / 'three-operand-experiments/data-1000'
OLD7 = SOURCE / 'benchmark-pilot/lora-7rules-fp16-retry'
MANIFEST7 = OLD7 / 'dataset/manifest.json'
BASELINE7 = OLD7 / 'evaluations/pretrained/validation'
A12 = SOURCE / 'benchmark-pilot/compare-3-vs-7/summary.json'
LOCAL = Path('/content/seven-rule-controls')
A3OUT = LOCAL / '07-seven-rule-operator-controls'

cmd = ['python','benchmark/operator_controls_7rules.py',
       '--manifest',str(MANIFEST7),'--data-dir',str(DATA),
       '--baseline',str(BASELINE7),'--historical-a12',str(A12),
       '--out',str(A3OUT)]
subprocess.run(cmd + ['--validate-only'], check=True)
subprocess.run(cmd, check=True)  # Zero GPU: verified saved predictions.
```

If the mount path is absent, recover those exact saved files from Drive before proceeding. Known source IDs: source manifest `1vByMvpzLM61LKKjDCnd_AhFpcQCuCIma`, predictions `1rw6axm_J7B88xIJorIKGNwfL7LHRgb7R`, config `1-pgct8EdA68Kb7yNJaXfJoGiAqAI84kN`, metrics `1YOOXQp4EN7crgtIQaRmnUSVgpbGjmob5`, answer tokens `1V1JZSfapUcWKzMKz9X3zBATUmNiSVloJ`. The shared `reconstruct_validation` helper can use existing partition assignments and the existing deterministic builder to reconstruct **only** validation sets; it labels this as reconstruction and refuses to certify an old baseline without full source-manifest identity. It never generates replacement test sets. No dataset replacement is automatic.

Inference is opt-in with `--allow-inference`, at most 343 forwards through the established evaluator. Do not use it when verified predictions are available. Aggregate-only recovery produces a clearly labeled historical summary without per-example CIs or matches. Identity mismatches are errors, not reasons to silently rerun.

## Console, outputs and IITM upload

Console output includes recovery status, workload, metrics, confidence intervals and interpretation. Files in `A3OUT` include `summary.json`, readable `summary.md`, plots, CSV and a compact verified validation/baseline cache used by experiment 08. No source images or model weights are copied.

Authenticate the **IITM account `24f2005474@ds.iitm.study.in`** for the API, even though GitHub and the read-only source mount use another account:

```python
from google.colab import auth
auth.authenticate_user()  # Select the IITM account in this separate API dialog.
!python benchmark/iitm_drive.py --verify-only
!python benchmark/iitm_drive.py --source "{A3OUT}" --experiment operator-controls
```

Destination is exactly folder `15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR`, with subfolder `07-seven-rule-operator-controls`. The uploader checks actual Drive API email and `canAddChildren` before writes, verifies uploaded IDs/sizes/MD5 hashes, resumes file updates with stable IDs and skips identical bytes. Wrong identity, domain restrictions, quota or write failures stop clearly; there is no old-Drive fallback. If cached API authentication selects the wrong account, use a fresh Colab authentication session and choose IITM; do not bypass the email check.

All important findings remain printed in the console and saved under `/content` **before upload**. If Drive is full or restricted:

```python
!python benchmark/iitm_drive.py --source "{A3OUT}" --manual-zip /content/07-seven-rule-operator-controls.zip
from google.colab import files
files.download('/content/07-seven-rule-operator-controls.zip')
```

This is a local ZIP, not a claimed upload. Manually upload it while signed into IITM at https://drive.google.com/drive/folders/15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR. No new upload was performed during implementation because the IITM OAuth account is not connected in the Codex tools.

## Limitations

Positive-only decisive-scene shortcuts, strong position bias, shared images and previously exposed test images limit reasoning claims. Bootstrap resamples sets to keep rotations together but does not account for shared images across sets. No causal inference about adding generating rules follows from comparing A3 to the old A1/A2 dataset.
