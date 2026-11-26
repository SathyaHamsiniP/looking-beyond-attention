# Shared files

Scripts: `build_sets.py`. The builder reads existing images and rule JSON; use the saved manifest for the current seven-rule experiments. `seven_rule_validation.py` checks validation inputs. `benchmark_paths.py` locates shared files. Install `requirements-seven-rule-controls.txt` for the seven-rule runners.

From the repository root:

```bash
python benchmark/code/00-common/build_sets.py --help
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
