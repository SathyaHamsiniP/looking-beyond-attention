# Seven-rule LoRA results

The aim of experiment [06 — LoRA](../../../../experiments/06-seven-rule-lora/README.md) is to see whether fine-tuning helps Qwen3.5-4B identify the generating Boolean rule of ten images, rather than simply preferring an answer position.

## Pretrained model

Before fine-tuning, Qwen got **48/343 validation answers correct (13.99%)** over 49 saved image sets and seven answer-option rotations. With seven choices, chance accuracy is 14.29%. This is the same pretrained evaluation used by [experiment 07 — operator controls](../operator-controls/README.md), where the saved individual predictions and answer-position analysis can be inspected.

The `pretrained-validation/metrics.json` file in this folder records the baseline metrics; the full individual baseline predictions are not duplicated here.

## Fine-tuned model

**A verified full-run evaluation has not yet been imported into this GitHub branch.** The current `partial-diagnostics/` folder records an earlier interrupted training attempt, not the result of a completed three-epoch LoRA run. The historical notebook claim of **116/343** is not a confirmed result without the corresponding final-model prediction records.

Once the completed experiment is imported and checked, this section should report the actual number of successful training updates, the chosen checkpoint and validation loss, accuracy on the same 343 validation presentations, a matched comparison with the pretrained predictions, per-rule performance and any remaining answer-position bias. Report only measurements present in the original results; do not derive new scores or plots for documentation.

## How to interpret the comparison

The 343 validation answers are **49 image sets presented in seven answer orders**, not 343 independent image sets. Improvement in validation accuracy would show better performance on this task, but by itself would not establish Boolean reasoning: the model could still rely on shortcuts or option preferences. The final test partition remains reserved.

[Experiment method and limitations](../../../../experiments/06-seven-rule-lora/README.md) · [Seven-rule benchmark overview](../../../../README.md)
