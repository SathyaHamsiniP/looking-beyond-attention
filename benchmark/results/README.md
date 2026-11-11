# Benchmark results

Results from the three-rule and seven-rule Qwen3.5-4B experiments are kept separate. Each link leads to the original measurements, figures, and a brief explanation.

| Rules | Rule identification | Internal probes | Activation patching | LoRA |
|---|---|---|---|---|
| Three | [Operator controls](three-rules/qwen3.5-4b/operator-controls/README.md) | [Probing](three-rules/qwen3.5-4b/probing/README.md) | [Patching](three-rules/qwen3.5-4b/activation-patching/README.md) | [LoRA](three-rules/qwen3.5-4b/lora/README.md) |
| Seven | [48/343 correct](seven-rules/qwen3.5-4b/operator-controls/README.md) | [Completed: up to 38.8% final-layer probe accuracy](seven-rules/qwen3.5-4b/probing/README.md) | [896 saved measurements](seven-rules/qwen3.5-4b/activation-patching/README.md) | [No verified full-run evaluation](seven-rules/qwen3.5-4b/lora/README.md) |

The new seven-rule probing run has 588 saved feature arrays with 588 matching metadata files, model predictions, probe prediction tables and two original plots. The complete [IITM Drive archive](https://drive.google.com/file/d/1sPEeKrm3q2nUDKYvpipe6xViIZ_Y41LQ/view) preserves the large arrays. The compact results belong in this repository; they should be imported byte-for-byte, without changing the saved scores or adding new experiments.

The three-rule and seven-rule setups are not identical, and not all experiments have final results. See the [benchmark overview](../README.md) and each experiment's README before comparing numbers.
