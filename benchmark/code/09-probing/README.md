# 09 — Probing

Scripts: `probing_7rules.py`. Uses the saved train/validation manifest, common rules/validation, the shared Qwen loader and one isolated import of the earlier `three-rules/probing.py` capture helper. Extraction saves features; `--fit-only` fits CPU probes after extraction.

From the repository root:

```bash
python benchmark/code/09-probing/probing_7rules.py --manifest /path/to/manifest.json --data-dir /path/to/data-1000 --out /content/seven-rule-probing --validate-only
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
