# 04 — Train Qwen3.5-4B to choose a Boolean rule

## What are we testing?

Does a little training help the model identify a rule from ten pictures? Does it change how activation patches affect its answers? We first measure the pretrained model. We then train small adapters and measure the same examples again.

**LoRA** adds small trainable matrices to model layers. The original model weights stay fixed. **QLoRA** also stores those original weights in 4-bit form to save memory. We use all supported language decoder layers. The vision encoder stays frozen. There is no single-layer training mode.

One input has ten actual images and the original rule-identification prompt. A is a green cube, B a blue sphere and C a red cylinder. The choices are R1 = `(A AND B) OR C`, R2 = `(A AND C) OR B`, and R3 = `(B AND C) OR A`, mapped to X/Y/Z. Training teaches the correct answer label, not the prompt text or ground-truth bit strings.

All pictures in a set satisfy its rule. Each set covers the five satisfying A/B/C configurations. A positive `001` reveals R1, `010` reveals R2, and `100` reveals R3. This shortcut is part of the original task. Better accuracy does not prove that the model evaluated the whole Boolean expression.

## Images and sets

Reuse the existing 1,000 pictures. Keep the original 800 training images. Split the 200 held-out pictures with seed 2026 into 100 validation and 100 final-test pictures. Each configuration has 12 or 13 pictures in each new held-out partition. No image crosses partitions.

| Partition | Image files | Ten-image sets | Sets per rule |
|---|---:|---:|---:|
| Training | 800 | 90 | 30 |
| Validation | 100 | 30 | 10 |
| Final test | 100 | 30 | 10 |

Image reuse between sets in the same partition is allowed and recorded. A set never repeats a file. Reordering the same ten files does not count as a new set. The builder checks each set by rebuilding it from the fixed seed, metadata and image checksums.

The training manifest records each answer mapping and target label. The correct letter is balanced across the 30 training sets for each rule. Evaluation uses all three cyclic mappings for every set: 30 sets × 3 rotations = 90 trials per partition. These are repeated measurements, not 90 independent sets.

## Code

- `build_training_sets.py` builds and checks the manifest without model weights.
- `train_lora.py` trains only decoder LoRA parameters and chooses the best validation checkpoint.
- `evaluate_lora.py` scores the pretrained model or a saved adapter on the same fixed sets.
- `compare_predictions.py` compares complete matched prediction files without loading a model.
- `common.py` holds small helpers. It reuses the original prompt, image processor adapter and decoder-layer finder from main. No older experiment file is edited.

The separate [patching guide](../05-post-lora-comparison/README.md) explains the before/after activation tests. Experiment 03 remains unchanged.

## Colab setup

Use a fresh GPU runtime, preferably a T4 with 16 GB of GPU memory. Ten-image training may still exceed that memory. Do the smoke test before a full run. No real GPU training has been verified in this implementation.

Mount Drive in a Python cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The remaining blocks are shell commands. Use `%%bash` at the top of a Colab cell. Repeat the variable block in each new shell cell, since shell variables do not persist between cells.

```bash
git clone --branch feature/04-simple-lora-patching https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
pip install -r three-operand-experiments/code/04-lora/requirements.txt
```

Colab provides CUDA PyTorch. If bitsandbytes was updated, restart the runtime when needed, then remount Drive. Optional fast kernels are not required. Do not install the old model's requirements into this runtime.

Set these paths. New output stays separate from the completed baseline and other branches:

```bash
cd /content/looking-beyond-attention
export CODE=three-operand-experiments/code
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000'
export RUNS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results/04-simple-lora'
export PATCH_RUNS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results/05-simple-lora-patching'
mkdir -p "$RUNS" "$PATCH_RUNS"
export MANIFEST="$RUNS/rule_identification_manifest.json"
```

If the dataset is already at `$DATA`, the downloader checks it without replacing it:

```bash
python "$CODE/00-data-generation/download_dataset.py" --output-dir "$DATA"
python "$CODE/04-lora/build_training_sets.py" --data-dir "$DATA" --manifest "$MANIFEST"
python "$CODE/04-lora/build_training_sets.py" --data-dir "$DATA" --manifest "$MANIFEST" --validate-only
```

These steps run on CPU. They should print 800/100/100 images and 90/30/30 sets. Never use the final-test results to choose training settings.

## Pin the model revision

Resolve the model's current commit once, then keep that file for every before/after run. This fetches metadata, not model weights. It needs Hugging Face access.

```bash
if [ ! -f "$RUNS/model_revision.txt" ]; then
  python -c 'from huggingface_hub import model_info; print(model_info("Qwen/Qwen3.5-4B").sha)' > "$RUNS/model_revision.txt"
fi
export REV="$(cat "$RUNS/model_revision.txt")"
```

The scripts require a full 40-character commit SHA. They reject `main`. Every model command below needs the same `$REV`, resolution and precision. Adapter metadata records these settings and rejects incompatible loading.

## GPU smoke test

This performs real forward and backward passes, but at most two optimizer steps on two training sets and two validation sets. It still supplies all ten images per input. Its outputs have a separate run ID and are not research results.

```bash
python "$CODE/04-lora/train_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$RUNS/training" --smoke-test
```

The model is still 4B parameters, even for this short test. If CUDA runs out of memory, the script reports memory use. Lower `--max-pixels` or `--rank` explicitly. Keep the same resolution for the pretrained evaluation, adapter evaluation and patching. Do not drop images. `--max-tokens` rejects long inputs; it does not truncate them. Gradient checkpointing and accumulation reduce memory use. Per-device batch size is fixed at one.

## Pretrained validation baseline

```bash
python "$CODE/04-lora/evaluate_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$RUNS/pretrained" --partition validation
```

The command prints its full results folder. Put that exact path in `BASELINE`. Do not use a smoke-test folder:

```bash
export BASELINE='/paste/the/printed/pretrained/validation/folder'
python "$CODE/04-lora/train_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$RUNS/training" --baseline-run "$BASELINE" --epochs 2
```

The defaults are NF4 4-bit weights, rank 8, alpha 16, dropout 0.05, AdamW at 2e-4, one input per device, accumulation over four inputs, and seed 2026. `--epochs 3` is supported. Validation runs every ten optimizer steps and at the end. It scores all three answer orders and does not update weights.

The code finds actual linear projections in each decoder block, including Qwen's different attention block types. It saves full target paths and verifies that every trainable parameter belongs to those LoRA modules. Base and vision parameters stay frozen. The loss uses only correct answer tokens with causal alignment. Multi-token answer labels use their whole conditional sequence, not just their first token.

## Checkpoints and held-out evaluation

Training saves `trainable.json`, `losses.jsonl`, `selection.json`, optimizer checkpoints and `best_adapter/`. Logs contain training loss, validation loss, learning rate and optimizer step. `selection.json` records the lowest validation loss and chosen step. No final-test example enters training or checkpoint selection.

To resume, repeat the same training command with `--resume /full/run/checkpoint-N`. The checkpoint stores adapter weights, optimizer, gradient scaler and random-number state. Load only trusted checkpoints. A changed configuration is rejected. Use the latest checkpoint; the script refuses one older than recorded losses.

Set `ADAPTER` to the `best_adapter` folder printed by training:

```bash
export ADAPTER='/paste/the/training/run/best_adapter'
python "$CODE/04-lora/evaluate_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$RUNS/finetuned" --adapter "$ADAPTER" --partition validation
```

Once settings are final, run both models on the final-test sets:

```bash
python "$CODE/04-lora/evaluate_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$RUNS/pretrained" --partition test
python "$CODE/04-lora/evaluate_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$RUNS/finetuned" --adapter "$ADAPTER" --partition test
```

Evaluation saves each set ID, image list, true rule, answer mapping, predicted rule, probabilities and correctness. It reports overall/per-rule/per-rotation accuracy, answer-label and rule counts, confusion matrices and chance accuracy of one third. The comparison script checks settings and every expected example key before writing tables and plots:

```bash
python "$CODE/04-lora/compare_predictions.py" --baseline /full/pretrained/test/run --finetuned /full/finetuned/test/run --output "$RUNS/prediction-comparison-test"
```

Use matching validation paths to inspect validation results first. Comparison output must be a new folder.

## Results and checks

No pretrained-model performance or GPU training results are available yet. CPU tests cover set construction, option targets, token loss, real PEFT on tiny random-weight Qwen3.5 models, frozen weights and adapter save/reload. They do not establish T4 fit or actual model accuracy.

```bash
python -m unittest discover -s three-operand-experiments/tests -v
```

Every model script supports `--validate-only` without loading weights and `--smoke-test` for a small real GPU check. Run folders use a hash of the configuration and include the full manifest, dataset fingerprint, model revision, package versions and code commit. Different settings get different folders. Exact partial evaluation/patching runs skip already saved keys. Do not change code or installed package versions in the middle of a resumable research run.

Do not commit adapters, checkpoints or the 1,000 pictures. Keep them on Drive. The completed Experiment 03 files are not outputs of this experiment and must not be overwritten.
