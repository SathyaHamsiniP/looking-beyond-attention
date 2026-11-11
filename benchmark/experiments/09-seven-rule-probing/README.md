# 09 — Seven-rule frozen probing

Can linear probes recover the generating rule from frozen Qwen representations, even when the model's answer is biased? **No seven-rule probing inference has been run yet.** The CPU validator has checked the actual saved manifest: 147 training sets and 49 validation sets, with disjoint source-image partitions. No test examples are extracted or scored.

## Method

A/B/C mean green cube, blue sphere and red cylinder; other objects are distractors. The unchanged [seven truth tables](../../code/00-common/rules_7.json) define three AND/OR rules, three XOR/OR rules and the nonempty, not-all-three rule. Each example contains positive images for one generating rule and seven numeric answer options.

Exhaustive truth-table enumeration establishes a minimum of **five distinct positive configurations** for every rule. Every selected subset is checked against all seven candidates:

| Images | Selection from each saved ten-image set | Meaning |
|---|---|---|
| 1 | Minimum SHA-256 of `2026|filename`, without consulting rule or bits | Deliberately ambiguous control; not a uniquely identifying task |
| 5 | First lexicographic original-index combination containing five distinct configurations compatible only with the generating rule | Oracle/logically informed minimal subset |
| 10 | Original ordered filenames unchanged | Original identifying task |

The five-image selection uses the true rule and truth tables; it is not a label-blind image-selection baseline. The old three-rule one-image condition used a decisive diagnostic scene, so these are different methodologies. No images or sets are regenerated. Selected names, bits, indices, compatible candidates and image/source hashes are saved in `input_report.json`.

Capture the final answer-prefix token at all 32 decoder-block outputs; freeze the base model. Rotation 0 is fixed by default, with numeric answers 1–7 and thinking disabled. Full extraction costs **588 forwards** (196 sets × three conditions); smoke costs three. Each set/condition's features and metadata are saved together. Their hashes and prediction records support interrupted runs. Resume rejects changed inputs/settings, duplicates, missing arrays and invalid shapes or non-finite values.

CPU fitting uses StandardScaler fitted only on the 147 training vectors, then LogisticRegression (`C=1`, `max_iter=2000`, seed 2026). Fit real labels and deterministic shuffled **training** labels at every layer/condition: 192 probes. Validation labels are used only for scoring. Chance is 1/7. Layer 32 is predeclared for confusion matrices; all 32 curves are descriptive, with no validation-selected “best layer” presented as independent evidence.

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

### Run the checks, extraction and probes

Use a fresh output folder with the moved code. If feature extraction started before this reorganization, resume or fit it with the [original commit](../../code/README.md#earlier-saved-runs); its saved source hashes must not be changed.

```python
OUT = Path('/content/seven-rule-probing')
base = ['python', 'benchmark/code/09-probing/probing_7rules.py', '--manifest', str(MANIFEST),
        '--data-dir', str(DATA)]
subprocess.run(base + ['--out', str(OUT), '--validate-only'], check=True) # CPU only
subprocess.run(base + ['--out', str(OUT)+'-smoke', '--smoke-test'], check=True) # GPU, separate output
subprocess.run(base + ['--out', str(OUT)], check=True) # GPU extraction, then CPU fits
# After interruption, retain all local outputs or restore the complete feature backup:
# subprocess.run(base + ['--out', str(OUT), '--resume'], check=True)
# Repeat just CPU fitting after complete extraction:
# subprocess.run(base + ['--out', str(OUT), '--fit-only'], check=True)
```

`config.json` records the extraction settings; `model_predictions.jsonl` records the frozen model readout. `features/` contains per-job arrays and metadata, outside Git. `probes/` contains layer accuracy/prediction/confusion/per-rule CSVs, `layer_accuracy.png`, `final_layer_confusions.png`, config and summary. Per-layer scores and the final summary print to the console before upload. Import only compact results with [the importer](../../code/tools/import_results.py); feature arrays stay outside Git.

### Save and upload results

Authenticate **inside a notebook Python cell**, not through `!python`:

```python
sys.path.insert(0, '/content/looking-beyond-attention/benchmark/code/tools')
import iitm_drive
service = iitm_drive.connect(authenticate=True)
iitm_drive.upload(service, OUT, 'probing')  # Compact outputs first.
# Preserve arrays outside Git for CPU fitting/resume; explicit larger backup:
iitm_drive.upload(service, OUT, 'probing', include_features=True)
```

Before writing, the API checks the actual account is `24f2005474@ds.study.iitm.ac.in` and that folder `15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR` is writable. Uploads verify size/hash and reuse remote file IDs. The subprocess CLI can reuse existing credentials, but never starts Colab's interactive authentication outside its notebook kernel. Outputs and important summaries remain local and in the console if Drive fills. Download a local backup instead:

```python
iitm_drive.manual_zip(OUT, Path('/content/probing-backup.zip'), include_features=True)
from google.colab import files
files.download('/content/probing-backup.zip')
```

## Limitations

Above-shuffle decoding would establish linearly accessible rule-correlated information under this protocol, not Boolean reasoning or causal use of that information. Rule-dependent oracle selection can itself create cues. One image cannot logically identify a rule; useful probe predictions there may exploit distributional correlations. Sets reuse images within a partition, so set counts are not independent image-level trials. Fixed rotation can mix answer position with rule identity. Validation is small (seven sets per rule), and examining 32 layers creates selection risk. These limitations apply even if the frozen model answers poorly. [Result status](../../results/seven-rules/qwen3.5-4b/probing/README.md).
