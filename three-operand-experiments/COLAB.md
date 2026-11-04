# Colab workflow

These commands are shell commands. In Colab, run each block in a `%%bash` cell. Mount Drive in a Python cell first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Choose a T4 GPU. Clone the feature branch once, then use the existing checkout:

```bash
git clone --branch feature/complete-research-pipeline https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
pip install -r three-operand-experiments/code/requirements-qwen35.txt
```

Colab must supply CUDA-compatible PyTorch. Restart the runtime after updating bitsandbytes if requested by the loader. Optional optimized attention kernels are not required. New workflows use Qwen/Qwen3.5-4B exclusively. Weights require Hugging Face access. For repeatable research, replace `--revision main` with a full model commit in **all** matching baseline/training/patching commands. Actual resolved revisions and package versions are logged. The CPU cloud setup does not provide CUDA.

Each shell cell needs these variables (or put them in one continuing shell script):

```bash
cd /content/looking-beyond-attention
export P=three-operand-experiments
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000'
export RESULTS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results'
export MANIFESTS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/task-manifests'
mkdir -p "$RESULTS" "$MANIFESTS"
```

## Dataset and tasks (CPU)

Reuse the existing dataset. The downloader validates it if already present; it never regenerates images.

```bash
python "$P/code/00-data-generation/download_dataset.py" --output-dir "$DATA"
for TASK in v1-positive-3rules v2-mixed-3rules v3-mixed-varied-rules; do
  python "$P/tasks/build_sets.py" --data-dir "$DATA" --task "$TASK" --output "$MANIFESTS/$TASK.json"
  python "$P/tasks/validate_sets.py" --data-dir "$DATA" --manifest "$MANIFESTS/$TASK.json"
done
export M="$MANIFESTS/v3-mixed-varied-rules.json"
```

A manifest cannot be overwritten with changed settings. Use a new filename when changing seeds or sizes. The final-test partition is for a single final evaluation after validation-based decisions.

## 01: preserved V1 controls and new task baseline

```bash
python "$P/code/01-operator-controls/operator_controls_qwen35.py" --data-dir "$DATA" --experiment-dir "$RESULTS" --experiment main_72 --precision 4bit
python "$P/code/01-operator-controls/evaluate_tasks.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --partition validation --shortcut-control
```

The second command is the pretrained V3 baseline required before training. For V2, pass its manifest instead. It reports per-rule and overall accuracy, chance (1/8 for V3), majority baseline, probabilities, and cyclic-order consistency. Historical caption/perception/gate controls are unchanged and are not automatically reinterpreted for mixed tasks. `--symbolic` is an explicit text-configurations control. `--shortcut-control` removes all distinguishing observations: remaining candidates and their probability mass are logged, with **no accuracy assigned to ambiguous inputs**.

## 02: concept and rule probing

New split-aware runners:

```bash
python "$P/code/02-probing/probe_tasks.py" --kind concepts --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --partition validation
python "$P/code/02-probing/probe_tasks.py" --kind rules --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --partition validation
```

Concepts use single images and eight A/B/C classes, with 800 train / 100 validation images. Rules use ten-image sets and fit only manifest training labels. Choices are omitted from rule-probe prompts. These fixed linear probes use scaled inputs, a shuffled-training-label control, all decoder layers, three pools, majority baseline and class counts. Probe accuracy measures information available to a classifier, not use when producing an answer. Select layers/settings on validation, then use `--partition test` once.

Historical dataset-v1 protocols are also preserved (800/200 concepts; 24/15 rule sets):

```bash
python "$P/code/02-probing/probe_concepts.py" --model qwen3_5 --data-dir "$DATA" --experiment-dir "$RESULTS"
python "$P/code/02-probing/probe_rules.py" --model qwen3_5 --data-dir "$DATA" --experiment-dir "$RESULTS"
```

Do not use the historical held-out aggregate to tune the new final-test protocol.

## 03: baseline patching

The running historical V1 experiment and its resume format are untouched. Do not restart, move, or alter its Drive files. Its existing command remains:

```bash
python "$P/code/03-activation-patching/patch_activations_qwen35.py" --data-dir "$DATA" --experiment-dir "$RESULTS" --pairs-per-rule 2 --rotations 3 --previous-token-control
```

For a **separate new V3 baseline**, use:

```bash
python "$P/code/03-activation-patching/patch_tasks.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --partition validation --previous-token-control
```

This deterministically chooses one receiver per rule with one image-disjoint donor for every candidate rule. If a requested plan has insufficient disjoint donors, generate more sets per rule under a new manifest filename. Pair plans use the same options for donor and receiver, and run all decoder layers and all cyclic rotations by default.

## 04: QLoRA

The baseline evaluator prints its full run directory. Set `BASELINE_EVAL` to that directory. It must contain a completed visual, pretrained evaluation on the same manifest, revision and image resolution. A smoke evaluation does not qualify.

```bash
export BASELINE_EVAL='/path/printed/by/evaluate_tasks.py'
python "$P/code/04-lora/train_lora.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --baseline-run "$BASELINE_EVAL" --rank 8 --alpha 16 --batch-size 1 --gradient-accumulation 4 --epochs 1 --validation-frequency 25 --checkpoint-interval 25
python "$P/code/04-lora/train_single_layer.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --baseline-run "$BASELINE_EVAL" --layers 16 --rank 8 --alpha 16
```

Each single-layer invocation loads a fresh base. Select any supported 1-based decoder layer, including hybrid linear-attention layers. `--layers 1 4 16` on `train_lora.py` targets a subset jointly. Training records actual full target paths and verifies that only selected LoRA parameters are trainable. Vision weights and base weights stay frozen.

Default 4-bit NF4 uses fp16 compute on T4. Decoder linear targets include attention, linear-attention projections and MLPs; they are discovered on the actual loaded model. Gradient checkpointing, disabled KV caching, accumulation and per-image pixel limits reduce memory use. Ten images are retained; no automatic image dropping or truncation occurs. If memory fails, explicitly lower `--max-pixels` and rerun **both** the baseline and training with the same value, or reduce rank / select fewer layers. T4 memory sufficiency for ten images is still unverified.

Training minimizes target-option token NLL. Option order varies deterministically per epoch/example, so targets follow the displayed rule. Validation loss on fixed validation orders selects `best_adapter`. Final-test labels never enter the optimizer or checkpoint selection. `--resume /full/run/checkpoint-N` restores adapter, optimizer, scaler, RNG, and the next optimizer boundary; use exactly the same settings. A configuration mismatch is rejected. Only load trusted local checkpoint state.

Set `ADAPTER` to the resulting `best_adapter` directory. Evaluate validation first:

```bash
export ADAPTER='/full/training/run/best_adapter'
python "$P/code/04-lora/evaluate_lora.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --adapter "$ADAPTER" --partition validation --shortcut-control
```

After selecting all settings, repeat the **pretrained evaluator and adapter evaluator** with `--partition test` on the identical manifest and full cyclic orders. Use full evaluation accuracy for before/after claims, rather than only the patching subset.

## 05: post-LoRA patching and comparisons

Use the identical patching flags for baseline and every adapter (including revision, precision, max-pixels, partition, selected layers, rotations and controls):

```bash
python "$P/code/05-post-lora-comparison/patch_finetuned.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --adapter "$ADAPTER" --partition validation --previous-token-control
```

Repeat for each selected single-layer adapter. Record each printed run directory. For final-test patching repeat both baseline and adapter patching with `--partition test` after validation decisions.

```bash
export BASE_PATCH='/full/new/baseline/patch/run'
export LORA_PATCH='/full/adapter/patch/run'
python "$P/code/05-post-lora-comparison/compare_results.py" --runs "$BASE_PATCH" "$LORA_PATCH" --output "$RESULTS/05-post-lora-comparison/qwen3_5/dataset-v1/v3-mixed-varied-rules/comparison-selected"
```

Add other adapter run directories to `--runs`. The output directory must be empty. Missing trials or mismatched plans, images, model revisions, labels or scoring settings are rejected. The primary metric is change in the receiver's true-rule probability in both conditions. Donor-rule probability has different target meanings for same/different conditions and is kept as a secondary raw measure. All original and patched probability vectors and correctness flags are saved. No filtering on model success is performed. Summaries include maxima and adjacent-layer changes; no statistical significance or unique reasoning layer is claimed.

Explicit legacy paths are accepted as inputs and inspected read-only, but a legacy V1 run is rejected as a matched V3 comparison. It lacks the new split/manifest plan. It can be analysed separately using its existing CSVs; rerun the new baseline on matched tasks for LoRA comparison. No migration tool runs automatically.

## Smoke tests and CPU checks

Every new model runner accepts `--validate-only` and `--smoke-test` (comparison uses completed files instead of a model). Validate first, then add `--smoke-test` to each model command before attempting full runs. Evaluation/extraction use two examples. Training uses at most two optimizer steps and two validation sets. Patching uses one same-rule and one different-rule pair, first/last layers, and retains the configured image count. Smoke outputs are configuration-separated and are not research results.

```bash
python -m unittest discover -s "$P/tests" -v
python "$P/code/03-activation-patching/test_patching.py"
python "$P/code/03-activation-patching/patch_tasks.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --validate-only
python "$P/code/04-lora/train_lora.py" --manifest "$M" --data-dir "$DATA" --results-root "$RESULTS" --baseline-run "$BASELINE_EVAL" --validate-only
```

## Exact output locations

Every new run prints its full path. The 16-character run ID hashes the configuration, manifest and run details; do not guess or manually reuse it. New results use these exact templates below `$RESULTS`:

| Runner | Relative directory |
|---|---|
| evaluate_tasks | `01-operator-controls/qwen3_5/dataset-v1/<task>/<run-id>/` |
| probe_tasks (kind recorded in config) | `02-probing/qwen3_5/dataset-v1/<task>/<run-id>/` |
| patch_tasks | `03-activation-patching/qwen3_5/dataset-v1/<task>/<run-id>/` |
| train/evaluate LoRA (variant and layers in config) | `04-lora/qwen3_5/dataset-v1/<task>/<run-id>/` |
| patch_finetuned | `05-post-lora-comparison/qwen3_5/dataset-v1/<task>/<run-id>/` |
| protected active historical patching | `activation_patching/qwen3_5/dataset-v1/pairs2_rot3_prev1/` |

New runs include config/run details, embedded task and split manifests, predictions or patch journals, summaries, CSVs/plots where appropriate, and adapters/checkpoints for training. Journals refuse duplicates. Changed settings get a different directory; exact settings resume safely. Large model checkpoints, task manifests and activations stay on Drive. The source repository retains historical archives without modifying them.

## Evidence status

Historical pilot findings are retained in the experiment READMEs and archived results. The user reports that the original Qwen3.5 patching smoke test passed in Colab and that the full baseline is in progress; its Drive outputs were not accessed here. New task generation, CPU invariants, and tiny random-weight Qwen3.5/PEFT tests can run without pretrained weights. These tests do not validate real-image 4-bit GPU training, T4 memory use, model answer-token IDs, or model accuracy. All new pretrained-model baselines, LoRA experiments and post-LoRA comparisons still require Colab smoke testing and real runs. No new research results are claimed.
