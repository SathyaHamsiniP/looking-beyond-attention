# Result tools

Scripts: `iitm_drive.py, import_results.py, lora_recovery.py`. Uses shared LoRA checks. Authenticate uploads inside a notebook: add `benchmark/code/tools` to `sys.path`, then call `iitm_drive.connect(authenticate=True)`. Importing copies compact results; recovery reads checkpoints or writes a separate backup. Neither modifies source results.

From the repository root:

```bash
python benchmark/code/tools/lora_recovery.py --root /path/to/saved-run
```

Results are written to the explicit `--out` path outside Git; tools use their explicit destination or backup path. See [code notes](../README.md) for old-run compatibility and [experiment descriptions](../../experiments/) for methods and findings.
