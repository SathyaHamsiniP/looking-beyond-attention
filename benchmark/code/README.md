# Benchmark code

Run commands from the repository root. The folders follow the experiment numbers:

- `00-common/`: dataset builder, rule definitions, shared validation and pinned requirements.
- `three-rules/`: earlier controls and probing, the shared Qwen model loader in `pilot_10_vs_20.py`, and the separate LoRA scaling runner.
- `06-lora/`, `07-operator-controls/`, `08-activation-patching/`, `09-probing/`: seven-rule runners.
- `tools/`: School Drive uploads, result imports and checkpoint backups.

There is one copy of each helper. `00-common/benchmark_paths.py` locates these folders so direct script commands and notebook imports use the same modules. Outputs belong outside the repository, normally under `/content`. Research descriptions remain in [experiments](../experiments/). Seven-rule measurements are indexed in [benchmark/results](../results/README.md); earlier three-rule measurements are indexed in [three-operand-experiments/results](../../three-operand-experiments/results/README.md).

## Earlier saved runs

Moving source files changes their bytes and therefore their source hashes. Rule JSON bytes, model/dataset hashes and all saved results are unchanged. New runs record the moved code's hashes, including the shared path helper. Config comparisons still reject an old run's checkpoint or features under different source code. Never edit saved configs or replace their hashes to make resume pass.

For the existing baseline and patching results, use the pre-move commit in a separate checkout. This keeps the old layout and checks exactly as recorded:

```python
# In Colab, after cloning feature/organize-benchmark into /content/looking-beyond-attention:
from pathlib import Path
import subprocess
ORIGINAL = Path('/content/7-rule-original')
if not ORIGINAL.exists():
    subprocess.run(['git','worktree','add','--detach',str(ORIGINAL),
                    '6d13d992432a9864b2879074f0b2e9d750bbdfa9'],check=True)
```

Use `ORIGINAL/'benchmark/operator_controls_7rules.py'` to check the existing 343 predictions, and `ORIGINAL/'benchmark/activation_patching_7rules.py'` to reproduce or resume the saved patching run. A recovered earlier LoRA checkpoint also requires its matching commit and package versions. The original result configs contain the exact code hashes; reject any checkout that does not match. This reorganization does not claim old runs were produced by the moved files.

For new probing runs, use the current `09-probing/probing_7rules.py`. Existing pre-move feature extractions must use the original checkout for resume or CPU fitting. A fresh LoRA run using the moved code needs a baseline produced by the same code and packages; copying the old baseline into that new run will correctly fail its config check. To reuse the old baseline without repeating inference, use its matching original runner for training as well.
