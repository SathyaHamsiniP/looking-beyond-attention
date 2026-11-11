# 08 — Activation patching

Scripts: `activation_patching_7rules.py`. Without `--run`, only validates inputs and prints the CPU plan. Uses common validation, the controls cache and the original `three-operand-experiments/code/03-activation-patching/patch_activations_qwen35.py` hooks. Saved pre-move runs require the original checkout.

From the repository root:

```bash
python benchmark/code/08-activation-patching/activation_patching_7rules.py --manifest /path/to/manifest.json --data-dir /path/to/data-1000 --cache /path/to/controls-cache --out /content/seven-rule-patching
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
