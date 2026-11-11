# Historical protocol documentation

Archived instructions: statuses and paths below describe earlier work. Use the current [benchmark index](../README.md) for experiment status and execution.

# Common Boolean-rule benchmark — draft foundation

**Status: provisional benchmark protocols and CPU-tested implementations. No seven-rule LoRA GPU results are claimed.** This folder is self-contained and imports no code from Experiments 01–04. The dataset builder and CPU tests use only Python's standard library. GPU inference requires PyTorch, Transformers, bitsandbytes, accelerate and Pillow.

## Files

- `rules.json` — our original three-rule pilot truth tables; **not** the final seven-rule benchmark.
- `build_sets.py` — reads existing images and constructs a reproducible JSON list of image sets. It does **not** render images, fine-tune, run inference or alter earlier experiments.
- `test_build_sets.py` — small CPU-only tests covering ambiguity, image splits and nested comparisons.
- `pilot_10_vs_20.py` — standalone pretrained Qwen3.5-4B paired operator-controls pilot (no imports from older experiments).

A **set** is one rule-identification question containing several images. The builder's `--counts` are sets **per rule**, not individual images.

## What the builder guarantees

- Every generated set contains each satisfying Boolean configuration at least once and has no repeated image.
- **Uniqueness among supplied candidate rules** is checked. For positive-only tasks, a rule is compatible with a set if it satisfies every image in that set. If more than one candidate is compatible, generation stops with an error.
- Images are disjoint between train/validation/test partitions. Images may appear in multiple sets **within** the same partition, and that reuse must be reported in any evaluation.
- With the same seed, a 10-image set is the prefix of its corresponding 20-image set, and a smaller training budget is a subset of a larger budget. Image set identity does not depend on image count.
- Rule truth tables and splits are **data**, not hidden in copied experiment code.

**Caution:** The existing rules are identifiable among R1/R2/R3 but could be satisfied by other Boolean formulas. A single decisive scene can reveal the rule; this is not yet a proof of reasoning across all images.

## Quick checks

From the `benchmark/` directory:

```bash
python -m unittest discover -p 'test_*.py'
python build_sets.py --help
```

To create **only a draft manifest**, supply a folder with `metadata.json` (records with `filename` and Boolean `bits`) and `images/`. The `--partitions` option accepts either a filename-to-partition JSON mapping or an object containing `image_partitions`. If omitted, the builder uses each record's existing `split`; it never invents splits.

Example with **illustrative, not agreed** counts (from the repository root):

```bash
python benchmark/build_sets.py \
  --data-dir /path/to/data-1000 \
  --partitions /path/to/agreed_partitions.json \
  --counts train=2 validation=1 test=1 \
  --images-per-set 10 \
  --out /path/to/draft_manifest.json
```

Output files are never silently overwritten if their content differs. Do not treat this preliminary three-rule generator as the finalized benchmark or use its drafts to claim matched results with a different protocol. When the protocol is agreed, use the same manifest for operator controls, probing, activation patching, and LoRA rather than creating four separate datasets.

## Running the 10 vs 20 pilot on Colab T4

Use the generated `pilot-10.json` and `pilot-20.json` manifests and the same `data-1000` folder. From the repo root:

```bash
python benchmark/pilot_10_vs_20.py --manifest-10 /path/to/pilot-10.json --manifest-20 /path/to/pilot-20.json --data-dir /path/to/data-1000 --out /path/to/results --validate-only
pip install 'transformers>=5.0' accelerate bitsandbytes pillow
python benchmark/pilot_10_vs_20.py --manifest-10 /path/to/pilot-10.json --manifest-20 /path/to/pilot-20.json --data-dir /path/to/data-1000 --out /path/to/results --smoke-test
python benchmark/pilot_10_vs_20.py --manifest-10 /path/to/pilot-10.json --manifest-20 /path/to/pilot-20.json --data-dir /path/to/data-1000 --out /path/to/results
```

The smoke test runs one 20-image forward pass without saving results. The full pilot runs 6 matched sets × 3 answer rotations × 2 sizes = 36 evaluations, saving `config.json`, incremental `predictions.jsonl`, and `summary.json` on Drive. It uses frozen pinned-revision Qwen3.5-4B with 4-bit NF4, not a fine-tuned adapter. This pilot uses the existing source `test` partition for diagnostics: **not an untouched final benchmark test**. Keep future final test examples locked. No LoRA or patching is run.

## Original-compatible operator controls (recommended; replaces the earlier exploratory pilot)

Run `operator_controls.py` rather than `pilot_10_vs_20.py` for the research comparison. It reproduces the original `main_72` setup with **8 sets/rule, 3 rules, and 3 cyclic label rotations**; the *only intended intervention* is changing the number of images per set, from 10 to 20. It recreates the original 10-image selections from the same `data-1000` training image pool, seed 2026, and appends 10 additional satisfying images. The original 10-image order is preserved. It uses the original prompt wording, 4-bit NF4 Qwen3.5-4B, and restricted next-token X/Y/Z softmax. The original model loader did not pin a specific revision; the new runner pins the revision previously recorded for Experiment 04 to make future runs reproducible. Compare results with original runs only after confirming the exact model revision and software environment.

```bash
python benchmark/operator_controls.py --data-dir /path/to/data-1000 --out /path/to/operator-controls-10-vs-20 --validate-only
python benchmark/operator_controls.py --data-dir /path/to/data-1000 --out /path/to/operator-controls-10-vs-20 --smoke-test
python benchmark/operator_controls.py --data-dir /path/to/data-1000 --out /path/to/operator-controls-10-vs-20
```

The full run uses **24 paired image sets × 3 option rotations × 2 image counts = 144 inferences**. It saves `config.json`, `paired_sets.json`, resumable `runs.jsonl`, and `summary.json`. It has no training, no validation-based model selection, and no need for train/test splits for the operator-controls task; it uses the original historical `train` image pool solely to reproduce that experiment's sampling. The same rule-identifiable sets are reused across rotations, so 144 is not the number of statistically independent examples. The new runner has not yet been executed on a GPU and must pass its smoke test first.

## Experiment 01: diagnostic image only (new variation)

This belongs to **01 — Operator Controls**, not a new numbered experiment. Run `benchmark/diagnostic_only.py` after completing the original-compatible 10-vs-20 comparison. It reads its `paired_sets.json`, `config.json`, and 10-image `runs.jsonl`; deterministically picks the *first logically unique diagnostic image* from each original 10-image set and gives Qwen only that image. Candidate rules, model revision, original prompt wording (apart from image count), X/Y/Z answer rotations and probability readout stay fixed. It writes to a **separate** output directory and never modifies reference results.

```bash
python benchmark/diagnostic_only.py --reference-dir /path/to/original-operator-controls --data-dir /path/to/data-1000 --out /path/to/diagnostic-only --validate-only
python benchmark/diagnostic_only.py --reference-dir /path/to/original-operator-controls --data-dir /path/to/data-1000 --out /path/to/diagnostic-only --smoke-test
python benchmark/diagnostic_only.py --reference-dir /path/to/original-operator-controls --data-dir /path/to/data-1000 --out /path/to/diagnostic-only
```

Expected **24 matched diagnostic scenes × 3 label rotations = 72 trials**. Results: `config.json`, `selected_images.json`, resumable `runs.jsonl`, `summary.json` (includes 1-vs-10 paired accuracy change, error flips, cluster-bootstrap interval). The diagnostic image is unique *among R1/R2/R3*. Note that the original 10-image coverage sets may contain **more than one** scene with the diagnostic Boolean configuration, and the selection rule intentionally chooses one; this comparison changes both image count and amount of redundant evidence. It does not repeat the original remove-diagnostic/remove-control ablation.

## Experiment 02 — matched 1-vs-10 rule probes (provisional)

`probing.py` is a self-contained frozen-Qwen3.5-4B probe pipeline, independent of historical Experiment 02. It consumes the ten-image manifest from `build_sets.py`, and selects the **first uniquely diagnostic** scene from each ten-image set for the paired one-image condition. **This does not use the old 24-set operator-controls evaluation as its probe training data.** The new manifest must provide separate `train` and `test` partitions with image-disjoint source images, at least three sets per rule in both. Candidate rules, splits and counts are explicitly recorded, so the original protocol's eventual benchmark choices require new rules/split/manifest artifacts, not rewriting extraction/scoring. Currently the runner supports exactly **three** candidate rules, matching the 1-vs-10 pilot; seven-rule tests require revising the model-answer mapping and identifiability checks later.

For a preliminary **illustrative** 30 train + 10 test sets per rule (not the original protocol-approved sample sizes):

```bash
python benchmark/build_sets.py \
  --data-dir /path/to/data-1000 \
  --counts train=30 test=10 --images-per-set 10 \
  --out /path/to/probe-manifest-10.json

python benchmark/probing.py \
  --manifest /path/to/probe-manifest-10.json \
  --data-dir /path/to/data-1000 \
  --out /path/to/probing-1-vs-10 --validate-only
```

Then in Colab T4 with `transformers>=5.0`, `torch`, `bitsandbytes`, `accelerate`, `numpy`, `pillow`, `scikit-learn` installed:

```bash
python benchmark/probing.py \
  --manifest /path/to/probe-manifest-10.json \
  --data-dir /path/to/data-1000 \
  --out /path/to/probing-1-vs-10 --smoke-test

python benchmark/probing.py \
  --manifest /path/to/probe-manifest-10.json \
  --data-dir /path/to/data-1000 \
  --out /path/to/probing-1-vs-10
```

The smoke test extracts final-token activations at every decoder layer for one matched pair; it saves nothing. The full run saves per-set/per-condition compressed activation files, resumable `model_predictions.jsonl`, `layer_accuracy.csv`, `probe_predictions.csv`, and `summary.json`. It fits a standardized logistic-regression probe at **each layer** for each input size, with true and shuffled train labels, then reports accuracy on held-out image-disjoint sets. No model fine-tuning is performed. The runner uses answer rotation 0 for Qwen's final predictions; do not interpret those answer accuracies as label-rotation-averaged results. The last-token representation is an answer-adjacent probe target; decoding rule identity does not demonstrate that the model computes Boolean logic. The one-image case can be solved by recognizing one diagnostic ABC configuration.

The inference code requires a GPU smoke test before a full run; this new probing runner has not yet been executed against a Qwen model. Results are exploratory until the final aligned rules, partitions and fixed evaluation protocol are chosen. **Never merge draft results into the previously completed experiments.**

## Seven generating rules: pretrained versus LoRA

`lora_7rules.py` implements a separate, positive-only **three-operand, seven-generating-rule** experiment. It includes R4–R7 as generating rules, unlike the completed R1–R3-under-3-vs-7-candidates comparison. It reuses `rules_7.json`, the tested set builder, and the comparison runner's model loader, numeric labels, option rotations and prompt. All runtime code is on this branch; no historical checkout is required. Historical code and completed results are unchanged.

This is a comparison with a different protocol's reported seven-rule, two-operand LoRA work, **not an exact replication**. His generation code is unavailable and his exact learning rate is unconfirmed. Differences cannot isolate operand count. His confirmed settings are Qwen3.5-4B, rank 8 and three epochs; our alpha, dropout, optimizer, resolution, split and checkpoint protocol are specified below, not attributed to him.

### Fixed data and protocol

Use only the existing 1,000 images and their metadata. The builder first looks for the supplied historical `rule_identification_manifest.json` (or the conventional path beside `data-1000`). It verifies its known canonical manifest hash and full dataset fingerprint, including image bytes, before reusing its exact `image_partitions`. This produces **800 training, 100 validation and 100 test source images**. The original metadata's 800/200 split alone is insufficient. If the historical file is unavailable, the seed-2026 procedure is reconstructed by Boolean configuration; provenance explicitly says it was reconstructed, not verified identical. A supplied but incompatible historical file is rejected.

| Partition | Sets per generating rule | Total ten-image sets | Full readout evaluations |
|---|---:|---:|---:|
| Training | 21 | 147 | Used for optimization only |
| Validation | 7 | 49 | 343 per model |
| Test | 14 | 98 | 686 per model |

Every set has ten distinct images, all satisfying its rule, with coverage of every satisfying configuration. Compatibility against all seven rules must identify exactly the intended rule. Duplicate unordered image sets and cross-partition image leakage are rejected. Image ordering is saved. Reuse within a partition is permitted and reported, including actual distinct images used and per-rule configuration counts. The real-data CPU audit during implementation verified all these invariants and exact historical partition reuse: **621/88/87 distinct images are used by the generated training/validation/test sets**, respectively. No manifest or result artifact is committed to Git.

Correct positions are fixed deterministically: each training rule has exactly three sets with each numeric answer position per epoch; validation checkpoint selection has exactly one of each position per rule. Seven cyclic rotations are used for all full validation/test readouts, preserving each set's image order. Sets, rotations and shared source images must not be treated as independent observations.

**The test images were previously exposed through exploratory experiments.** This is not an untouched final benchmark. Positive-only diagnostic scenes may support shortcut learning, and evaluation does not establish unseen Boolean-configuration generalization or prove logical reasoning.

### Model, loss and selection

The model is `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, using 4-bit NF4, double quantization, FP16 compute, SDPA, thinking disabled and `128*28*28` maximum pixels per image. Ordered RGB resizing and chat preprocessing match `compare_3_vs_7.py`. Every input has ten images and the same seven-choice prompt. No true-rule identifier, Boolean metadata or answer is included as extra input. Inputs are never silently truncated or reduced; a 16,000-token guard fails explicitly.

Actual tokenizer representations are checked for distinct single-token numeric labels 1–7, including append-only behavior at the rendered assistant answer boundary. Training uses full-vocabulary cross entropy on the next answer token; prompt/image tokens are not loss targets. Inference uses the same token IDs and next-token logits, normalized over the seven numeric labels. Restricted-choice NLL and checkpoint-selection full-vocabulary loss are different quantities and are labeled separately.

Training starts from the pretrained base with seed 2026 and fresh rank-8 LoRA adapters, alpha 16, dropout 0.05, AdamW learning rate `5e-5`, weight decay 0.01, betas (0.9, 0.999), epsilon `1e-8`, microbatch 1, accumulation 4, gradient clip 1 and initial AMP scale 8. The vision tower and base weights are frozen. All supported decoder linear targets are discovered and saved; the target count is not assumed to be 248. Non-reentrant gradient checkpointing is used.

Three epochs give **441 set presentations and 111 planned updates**: 36 batches of four plus one batch of three per epoch. The partial batch is normalized by three and is not padded. An AMP-skipped update caused by non-finite gradients is replayed on the same batch with the reduced scale, with at most **four retries after the initial attempt** (five attempts total). Invalid gradients are cleared and all forward/backward passes are recomputed. Python, NumPy, CPU Torch and CUDA RNG states are restored to the batch-start snapshot before each replay, so dropout and other random draws match; GradScaler retains its reduced scale. A successful update requires an actual optimizer call, finite gradients and changed LoRA B weights. Any unsafe mutation, missing scale backoff or persistent failure aborts without advancing the update cursor or saving an optimizer checkpoint. The 441 scheduled presentations are unchanged; additional retry presentations are recorded separately. Resume restores the preceding successful checkpoint, optimizer/scaler, complete RNG state and exact data order.

At each epoch end, selection evaluates **49 fixed, position-balanced validation inputs**. The adapter with minimum full-vocabulary answer-token loss is selected. Thus checkpoint selection adds **147 validation-loss forwards** across training, rather than a seven-rotation sweep after every epoch. Final selected-adapter validation uses all 343 rotations, matched to the pretrained validation baseline. Test performance never selects an epoch or hyperparameter.

### Colab execution

From the repository root, fetch the existing branch without overwriting local work:

```python
!git fetch origin feature/benchmark-foundation
!git switch feature/benchmark-foundation
!git pull --ff-only origin feature/benchmark-foundation
!pip install 'transformers>=5.0' peft bitsandbytes accelerate pillow numpy matplotlib
```

Use your existing `DATA` and `PILOT` Python variables. `DATA` must point to `three-operand-experiments/data-1000`. The root below must be new; keep the Python/package environment unchanged between baseline, training and evaluation, because configuration identity is checked on resume and comparison.

```python
from pathlib import Path
import subprocess

OUT7 = Path(PILOT) / 'lora-7rules'
HISTORICAL = Path(DATA).parent / 'results/04-simple-lora/rule_identification_manifest.json'
cmd7 = ['python', 'benchmark/lora_7rules.py']
args7 = ['--data-dir', str(DATA), '--out', str(OUT7)]

def run7(action, *extra):
    subprocess.run(cmd7 + [action] + args7 + list(extra), check=True)

# CPU check, no saved manifest/results and no model loading.
run7('build', '--historical-manifest', str(HISTORICAL), '--validate-only')

# Save new Drive artifacts; refuses an existing experiment root.
run7('build', '--historical-manifest', str(HISTORICAL))
run7('baseline', '--validate-only')  # Revalidate saved manifest and source data.

# Full seven-generating-rule pretrained validation: 49 sets x 7 = 343 forwards.
run7('baseline')

# Real GPU forward/backward/update check, in a separate smoke directory.
# One update of four training examples plus seven smoke validation forwards.
run7('smoke')

# Train 3 epochs; checkpoint after each successful optimizer boundary.
run7('train')
# After interruption, rerun this instead of the preceding line:
# run7('train', '--resume')

# Evaluate the selected epoch on the identical full validation set.
run7('evaluate')
run7('compare')
```

Evaluation is also incremental: use `run7('baseline', '--resume')` or `run7('evaluate', '--resume')` after interruption. A completed compatible evaluation can be reopened with `--resume` without loading the model. No incomplete or incompatible output is silently reused. Optimizer checkpoint files are trusted local artifacts only; do not load untrusted `training_state.pt` files.

Only after selection and validation comparison are complete, explicitly freeze the adapter and evaluate the previously exposed test partition:

```python
run7('final-test')  # 686 pretrained + 686 selected-adapter evaluations.
# After interruption: run7('final-test', '--resume')
```

This writes `selection_freeze.json` before test inference. Any later training attempt under that experiment root is refused. The selected-adapter identity, model settings and manifest must continue matching the freeze record. A new root is required for a new experiment.

### Outputs and validation status

All artifacts are below `{PILOT}/lora-7rules/`, outside Git:

- `dataset/`: `manifest.json`, `partitions.json`, `validation_report.json`; includes source/partition hashes, seed, filenames, configurations, mappings and reuse statistics.
- `evaluations/pretrained/validation/` and `evaluations/finetuned/validation/`: config, actual answer token IDs, incremental `predictions.jsonl`, and `metrics.json`.
- `smoke-training/`: isolated update evidence/checkpoint, never accepted as a completed scientific model.
- `training/`: config, actual token IDs, resolved trainable-module inventory, `losses.json` (including retry counts/scales and per-attempt diagnostics), `update_attempts.jsonl` (append-only diagnostics, not a training cursor), `update_failure.json` on exhausted/unsafe updates, `validation_losses.json`, `latest/` adapter plus optimizer/scaler/RNG state, `best_adapter/`, `selection.json`, and `complete.json`. The selected epoch is explicit. Only latest and best adapters are retained; interrupted directory replacement can recover the preceding complete checkpoint.
- `final-test/{pretrained,finetuned}/test/`: resumable paired test predictions and metrics.
- `comparison/{validation,test}/`: `summary.json` and `loss_curves.png`; reports accuracy/chance adjustment, per-rule confusion, position counts, correct-rule probability, restricted-choice NLL, rotation consistency and paired accuracy change.

Run CPU tests with:

```bash
python -m unittest discover -s benchmark -p 'test_lora_7rules.py' -v
```

Tests cover truth tables, partitions, set counts, coverage, identifiability, deterministic generation, label balance, exact preprocessing parity, answer-token gradients, partial accumulation, frozen weights, genuine tiny-model CPU optimizer updates, checkpoint state/RNG recovery, incompatible output refusal and paired evaluation mappings. CPU tests do not establish actual Qwen tokenizer compatibility, 4-bit T4 memory capacity or GPU training success; the real GPU smoke test is required. **No full seven-rule GPU training or inference was run during implementation.**

### Retrying the initial FP16 gradient-scaling failure

The retry limit is four replays, not four successful updates. For example, if all attempts overflow starting at scale 8, attempted scales are 8, 4, 2, 1 and 0.5. Each attempt records whether the optimizer ran, the scale before/after, non-finite gradient-value counts and affected parameter names/counts, plus changed LoRA B matrices. Successful update records embed their attempts; failed attempts never enter the checkpoint's successful-update journal. On exhaustion, gradients are cleared, batch-start RNG is restored and detailed diagnostics are preserved. Resuming replays from the last successful checkpoint; diagnostic journal entries may repeat after an interruption and must not be counted as optimizer updates.

Updating this implementation changes the recorded source hash. **Do not use `--resume` against the old failed smoke directory or edit its config to bypass checks.** Preserve it and copy only its unchanged dataset artifacts into a fresh experiment root:

```python
!git fetch origin feature/benchmark-foundation
!git switch feature/benchmark-foundation
!git pull --ff-only origin feature/benchmark-foundation

from pathlib import Path
import shutil
import subprocess

OLD7 = Path(PILOT) / 'lora-7rules'
RETRY7 = Path(PILOT) / 'lora-7rules-fp16-retry'
# Refuses an existing destination; choose a fresh suffix if already used.
shutil.copytree(OLD7 / 'dataset', RETRY7 / 'dataset')
smoke7 = ['python', 'benchmark/lora_7rules.py', 'smoke',
          '--data-dir', str(DATA), '--out', str(RETRY7)]
subprocess.run(smoke7 + ['--validate-only'], check=True)
subprocess.run(smoke7, check=True)
# Only if this new-version smoke run is interrupted:
# subprocess.run(smoke7 + ['--resume'], check=True)
```

This reuses the exact saved image sets and partitions and leaves the original failure intact. It changes no rules, model settings, learning rate, rank, epochs, objective or target modules. For later training under the new root, run the matching new-version baseline first; use `--resume` only for checkpoints created with the same code/configuration. CPU tests simulate native GradScaler overflow/backoff and verify recovery against a clean reduced-scale update, persistent-failure bounds, gradient cleanup and RNG replay. They do not establish real Qwen/T4 recovery; run the GPU smoke test above before full training.

## 07/08 — Seven-rule controls and activation patching

- [07 — A3 operator controls](experiments/07-seven-rule-operator-controls/README.md): zero-GPU reuse of verified 343 saved validation predictions; A1/A2 are not rerun.
- [08 — Activation patching](experiments/08-seven-rule-activation-patching/README.md): bounded same/different-rule controls using the existing historical hooks and a seven-label scorer.

Work locally in Colab `/content`. All new uploads use verified IITM account `24f2005474@ds.study.iitm.ac.in` and folder `15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR` through `iitm_drive.py`; never the full original Drive. Key findings and progress print in the console before upload, and local ZIP fallback remains available if Drive is full. See the two guides for exact fresh-runtime commands, recovered-source IDs, pinned dependencies and workload estimates. No expensive GPU runs or new Drive uploads were performed during implementation.
