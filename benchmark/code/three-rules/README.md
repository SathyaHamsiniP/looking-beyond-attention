# Three-rule experiments

Scripts: `operator_controls.py, diagnostic_only.py, compare_3_vs_7.py, pilot_10_vs_20.py, probing.py, lora_scaling.py`. Uses the common dataset builder and rules. `pilot_10_vs_20.py` also supplies the shared pinned Qwen loader; it is not duplicated. Scaling uses its own pinned historical checkout.

From the repository root:

```bash
python benchmark/code/three-rules/probing.py --help
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
