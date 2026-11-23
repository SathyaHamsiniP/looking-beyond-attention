# CPU tests

From the repository root:

```bash
python -m unittest discover -s benchmark/tests -p 'test_*.py'
```

Tests cover the dataset builder, three/seven-choice controls, LoRA, controls and patching, probing, saved results and result tools. The earlier local scaling runner and its eight tests are included, preserving all 63 existing tests. Three additional tests check moved imports, command-line validation and source/path compatibility. Four more check the completed seven-rule LoRA records, matching validation examples, checkpoint choice and original file hashes. Tests build temporary fixture datasets; they do not regenerate the research dataset or run a pretrained GPU model. `setup_paths.py` enables the same shared imports as the scripts.
