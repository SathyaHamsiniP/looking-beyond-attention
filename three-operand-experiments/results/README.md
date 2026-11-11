# Three-operand experiment results

These are the earlier three-rule results. They belong here rather than in the seven-rule [benchmark](../../benchmark/results/README.md). Different runs are kept distinct, even when their experiment names sound similar.

- [01 — Operator controls](01-operator-controls/): original Qwen3-VL and Qwen3.5 controls, plus separate Qwen3.5 [10/20-image](01-operator-controls/qwen3.5-4b/ten-vs-twenty-images/summary.json) and [three/seven-choice](01-operator-controls/qwen3.5-4b/three-vs-seven-choices/summary.json) benchmarks
- [02 — Probing](02-probing/qwen3.5-4b/README.md): later three-rule layer scores and predictions; the small earlier concept probe is described [here](../experiments/02-probing/README.md)
- [03 — Activation patching](03-activation-patching/qwen3.5-4b/README.md): 3,456 saved interventions, with their raw data and plots kept once in the original analysis folder
- [04 — LoRA](04-lora/qwen3.5-4b/README.md): matched validation predictions, training records and comparison graphs

The original 48-image experiments, later three-rule pilots and seven-rule benchmark used different settings. Their accuracies are not matched estimates of a rule-count effect.
