# 06 — LoRA

Scripts: `lora_7rules.py`. Reads `OUT/dataset/manifest.json` and `partitions.json`; uses the common rules/builder and three-rule folder's model loader. Training also requires a matching pretrained baseline. Old checkpoints require their original code; see the compatibility notes in the parent README.

From the repository root:

```bash
python benchmark/code/06-lora/lora_7rules.py train --data-dir /path/to/data-1000 --out /content/seven-rule-lora --validate-only
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
