# Compact verified results

Each family has separate `operator-controls`, `probing`, `activation-patching` and `lora` directories under `qwen3.5-4b`. No Qwen3-VL outputs are included. Original result locations are preserved. Import receipts record source URLs, sizes and SHA-256; numerical checks are recorded in provenance. Image datasets, arrays, adapters and training-state binaries stay outside Git.

| Family | Controls | Probing | Patching | LoRA |
|---|---|---|---|---|
| Three | [10/20 pilot](three-rules/qwen3.5-4b/operator-controls/README.md) | [Historical test results](three-rules/qwen3.5-4b/probing/README.md) | [3,456 rows](three-rules/qwen3.5-4b/activation-patching/README.md) | [29/90 → 30/90 validation](three-rules/qwen3.5-4b/lora/README.md) |
| Seven | [48/343 validation](seven-rules/qwen3.5-4b/operator-controls/README.md) | [Not run](seven-rules/qwen3.5-4b/probing/README.md) | [896 rows](seven-rules/qwen3.5-4b/activation-patching/README.md) | [Partial artifacts only](seven-rules/qwen3.5-4b/lora/README.md) |

Different splits, image budgets, answer choices, inference settings and training budgets prevent a matched cross-family comparison. Recompute from the raw files before interpreting aggregates. See the [benchmark index](../README.md) and experiment READMEs for these methodological differences.
