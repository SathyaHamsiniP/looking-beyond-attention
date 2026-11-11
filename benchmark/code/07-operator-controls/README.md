# 07 — Operator controls

Scripts: `operator_controls_7rules.py`. Uses common seven-rule validation and the saved pretrained predictions. For the published pre-move baseline, use its original checkout as explained in the parent README.

From the repository root:

```bash
python benchmark/code/07-operator-controls/operator_controls_7rules.py --manifest /path/to/manifest.json --data-dir /path/to/data-1000 --baseline /path/to/matching-baseline --out /content/seven-rule-controls --validate-only
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
