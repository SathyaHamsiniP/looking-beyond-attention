# 05 — Post-LoRA activation patching and comparison

## Goal and method

Compare causal output sensitivity before and after adapter training on matched inputs. Load the base or adapter, record donor/receiver predictions, then replace a token's decoder-block output at one layer. Use the same pair plan, option rotations, layers, image resolution and scoring for every variant. Include same-rule donors and optional previous-token controls.

The primary outcome is change in probability of the receiver's true rule, defined identically in both conditions. Raw donor-rule changes and full probability vectors remain available. We include every selected receiver regardless of whether either model is correct. Peak effects and adjacent-layer changes are descriptive, not proof of a unique reasoning layer or statistical significance.

## Data and commands

Use dataset-v1 V3 validation sets for development, then matched final-test sets after settings are fixed. See [COLAB.md](../../COLAB.md#05-post-lora-patching-and-comparisons) for exact baseline, adapter, single-layer variant and comparison commands. Run smoke tests first.

## Outputs

A matched pair plan; raw original/patched probabilities, predictions and correctness; layer-effect CSV/plots; descriptive summaries; a cross-variant comparison CSV/plot and receiver-subset accuracy. Use full Experiment 01/04 evaluation for overall before/after accuracy.

## Results and limitations

No new model patching results are available. CPU tests validate matching and hook mechanics. GPU adapter loading and actual patching require Colab tests. Explicit legacy baseline paths are read-only inputs, and incompatible V1/V3 protocols are rejected rather than described as matched. The active historical Experiment 03 directory is never changed or migrated.
