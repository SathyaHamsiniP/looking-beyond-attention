# 03 — Activation patching

## Goal

Find which Qwen3.5-4B layers change the model's answer to a Boolean rule question.

We use the [1,000-image dataset](https://github.com/SathyaHamsiniP/looking-beyond-attention/releases/tag/dataset-v1). The model sees **10 images at a time**. We use only the 200 held-out images for this experiment. No model weights are changed.

## What we do

Take two different sets of ten images:

- **Same rule:** Both sets have the same rule, such as R1 and R1. This is a control.
- **Different rules:** The sets have different rules, such as R1 and R2. We test whether the answer moves toward R2.

We first run each set normally. Then, at **one decoder layer at a time**, we replace the final token's activations for the first set with those from the second set. We run the model again and record the change in its answer.

The script tests **every layer**, not every third layer. Layer numbers start at **1**. It also records the unchanged model's answer for both sets.

For an extra check, `--previous-token-control` patches the token immediately before the last one. With `--rotations 3`, we repeat the test with all three cyclic answer-label orders. This helps reveal answer-position bias.

## Run on Colab

Use a **T4 GPU** and mount Google Drive. Clone or pull this repository.

Install requirements:

```bash
pip install -r three-operand-experiments/code/requirements-qwen35.txt matplotlib
```

The following commands assume that you already have the images at:

```text
/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000
```

Use your own path if it is different.

**Optional CPU checks (no Qwen model):**

```bash
python three-operand-experiments/code/03-activation-patching/test_patching.py
```

**1. Check the data and pairs (no GPU inference):**

```bash
python three-operand-experiments/code/03-activation-patching/patch_activations_qwen35.py \
  --data-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000" \
  --validate-only
```

**2. Small GPU test:**

```bash
python three-operand-experiments/code/03-activation-patching/patch_activations_qwen35.py \
  --data-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000" \
  --smoke-test
```

**3. Main run, including the extra checks:**

```bash
python three-operand-experiments/code/03-activation-patching/patch_activations_qwen35.py \
  --data-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000" \
  --pairs-per-rule 2 --rotations 3 --previous-token-control
```

The main command uses **6 receiving sets** (2 for each rule). Each receiver is paired with **one same-rule donor** and **two different-rule donors**, making **18 pairs**. Three answer orders and two token positions are tested at every layer. The script saves each result as it finishes, so you can rerun the same command if Colab disconnects.

For a smaller run, omit `--previous-token-control` and use `--rotations 1`. To test up to five receivers per rule, set `--pairs-per-rule 5`.

**Note:** A full run with ten images per input and multiple controls is computationally heavy. Run the small test first. Qwen3.5-4B still needs to fit in Colab GPU memory.

## What gets saved

Results go to your existing Google Drive results folder, under:

```text
results/activation_patching/qwen3_5/dataset-v1/
```

Each set of run settings has its own folder containing:

- `patches.csv` — One row per pair, answer order, layer and token position. Includes the exact layer number, predictions and probability changes.
- `layer_effects.csv` — Average change at each layer for same-rule and different-rule pairs.
- `layer_effects.png` — Graph of the changes across layers.
- `summary.json` — Layer with the highest effect, layer with the largest step, and first layer that passes the control cutoff.
- `config.json` — Settings and image pairs used.

The first clear effect uses a **0.05 probability-point difference from the controls**. This is a chosen cutoff, **not** a statistical significance test. We also report the largest adjacent-layer increase and the highest overall effect separately.

## What the results mean

A large effect at a layer means changing its activation can change the final answer. It does **not** prove that the layer alone performs Boolean reasoning.

Comparing same-rule and different-rule patches helps us see whether the effect is related to the rule or simply caused by changing the image set. The previous-token and answer-order checks add further controls.

**Status:** The user reports that the Colab smoke test passed and the full baseline run is in progress. Those Drive outputs have not been inspected in this task.

## Reference

We used [Roshan's two-operand patching experiment](https://github.com/RoshanYSingh23/looking-beyond-attention/blob/main/code/experiments/exp7_patching.py) as a reference. Our code is for Qwen3.5-4B, three rules and the new 1,000-image dataset.

## New manifest workflow

The historical code, data and results above are preserved. New task versions use separate runners and configuration-based directories. See [the Colab runbook](../../COLAB.md) for exact commands, output locations, split definitions and smoke tests. New runs have no pretrained-model results yet.

The user reports a successful Colab smoke test and a full baseline run in progress. Its results were not accessed here. `patch_activations_qwen35.py` and `results/activation_patching/qwen3_5/dataset-v1/` remain unchanged. `patch_tasks.py` supplies future matched manifest experiments in a separate numbered directory.
