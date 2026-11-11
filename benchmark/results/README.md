# Experiment results

The folders below separate three-rule and seven-rule Qwen3.5-4B experiments. Each contains operator controls, probing, activation patching and LoRA. Qwen3-VL results remain in their original locations and are not mixed here.

| Rules | Operator controls | Probing | Activation patching | LoRA |
|---|---|---|---|---|
| Three | [10/20 images and three/seven choices](three-rules/qwen3.5-4b/operator-controls/README.md) | [Earlier test results](three-rules/qwen3.5-4b/probing/README.md) | [3,456 measurements](three-rules/qwen3.5-4b/activation-patching/README.md) | [29/90 → 30/90 validation](three-rules/qwen3.5-4b/lora/README.md) |
| Seven | [48/343 validation](seven-rules/qwen3.5-4b/operator-controls/README.md) | [Not yet run on GPU](seven-rules/qwen3.5-4b/probing/README.md) | [896 measurements](seven-rules/qwen3.5-4b/activation-patching/README.md) | [Partial training records](seven-rules/qwen3.5-4b/lora/README.md) |

The datasets, image counts, answer choices, model settings and evaluation methods differ. See the [overview](../README.md) before comparing scores across experiments.

Results were copied without changing the originals. Each `provenance.json` records where its files came from and the checks performed; see the overview for details. Source images, activation arrays, adapters and optimizer state remain outside GitHub. Missing experiments have no performance scores here.
