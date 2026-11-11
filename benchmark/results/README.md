# Seven-rule benchmark results

This folder contains the results for the seven-rule experiments with Qwen3.5-4B. Earlier three-rule work is kept in [three-operand-experiments](../../three-operand-experiments/results/README.md), so the same results do not appear twice.

| Experiment | Saved results |
|---|---|
| [07 — Operator controls](../experiments/07-seven-rule-operator-controls/README.md) | [48/343 correct; answer-position analysis](seven-rules/qwen3.5-4b/operator-controls/README.md) |
| [08 — Activation patching](../experiments/08-seven-rule-activation-patching/README.md) | [896 interventions and layer effects](seven-rules/qwen3.5-4b/activation-patching/README.md) |
| [09 — Probing](../experiments/09-seven-rule-probing/README.md) | [Original predictions, CSVs and plots](seven-rules/qwen3.5-4b/probing/README.md) |
| [06 — LoRA](../experiments/06-seven-rule-lora/README.md) | [Pretrained baseline and partial training records](seven-rules/qwen3.5-4b/lora/README.md); no verified completed fine-tuned result |

The full [seven-rule probing archive](https://drive.google.com/file/d/1sPEeKrm3q2nUDKYvpipe6xViIZ_Y41LQ/view) keeps the large frozen features outside GitHub. See the [benchmark overview](../README.md) before comparing scores across experiments.
