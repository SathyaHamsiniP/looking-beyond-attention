# 03 — Activation patching

**Status: Planned. No runs or results yet.**

## Goal

Find which model layers affect the final answer.

In activation patching, we take saved activations from one input and replace the activations of another input at one layer. We check how much the answer changes.

## Planned test

1. Choose two ten-image sets with different rules.
2. Run both through the model without changing its weights.
3. Replace the receiving model's final-token activation with the donor's activation at one layer.
4. Repeat at each layer. Keep the same image sets and settings.
5. Check whether the donor rule becomes more likely.

Also check two controls: patch from another set with the same rule, and patch at a position other than the final token.

## What to save

Save one row per pair, layer and control. Include the original answer, new answer and change in probability for the donor rule.

Report these separately:

- **First clear effect:** The earliest layer where patching changes the answer beyond the controls. Set the cutoff before running.
- **Largest step:** The biggest increase in effect from one tested layer to the next.
- **Highest effect:** The layer where patching changes the probability the most.

Save a layer-by-layer CSV, a plot and a short summary. Show the layer numbers and the size of each change. Use every layer when finding the largest step. If only some layers are tested, say so.

## Next comparisons

**A. Full-model LoRA:** Train LoRA adapters across the model using training images. Use held-out examples that the tuned model answers correctly. Repeat patching and compare the layers with the untuned model.

**B. One-layer LoRA:** Tune adapters at one chosen layer at a time. Use the same training data and training settings for each layer. Measure held-out accuracy and patching effects. Check whether the layer with the largest effect changes.

Use the same rule task, test data, scoring and layer numbering across runs. Keep training and test images separate. Do not pick and report a layer using only the test data.

## Data

The plan is to use the new 1,000-image dataset for these tests. The experiment code is not ready yet. The older probing and sampling scripts must first be updated for the new data.

## Reference

Roshan's [patching code](https://github.com/RoshanYSingh23/looking-beyond-attention/blob/main/code/experiments/exp7_patching.py) is for the older two-operand task. His run checks selected layers and saves layer-by-layer probability changes. It reports the layer with the highest effect, not a separate largest-step summary.

We will use it as a reference, not treat its results as results from our three-operand task.
