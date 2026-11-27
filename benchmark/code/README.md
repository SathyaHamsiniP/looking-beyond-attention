# Benchmark code

Run commands from the repository root. The folders follow the experiment numbers:

- `00-common/`: dataset builder, rule definitions, shared validation and pinned requirements.
- `three-rules/`: earlier controls and probing, the shared Qwen model loader in `pilot_10_vs_20.py`, and the separate LoRA scaling runner.
- `06-lora/`, `07-operator-controls/`, `08-activation-patching/`, `09-probing/`: seven-rule runners.
- `tools/`: Optional archive uploads, result imports and checkpoint backups.

There is one copy of each helper. `00-common/benchmark_paths.py` locates these folders so direct script commands and notebook imports use the same modules. Outputs belong outside the repository under `/content`. Research descriptions remain in [experiments](../experiments/). Seven-rule measurements are indexed in [benchmark/results](../results/README.md); earlier three-rule measurements are indexed in [three-operand-experiments/results](../../three-operand-experiments/results/README.md).

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

## Run Experiments 11 and 12 in Colab

Start on a CPU runtime. Use a persistent folder outside the checkout for `WORK`. Set `RENDERER` to your existing renderer checkout at revision `71ee6120e129bd466f5cc99745a20060e61403b6`. Set `ADAPTER` to the recovered seven-rule epoch-3 `best_adapter` directory. These are shell commands for a Colab `%%bash` cell. Set the four paths with Colab `%env` first so they survive across cells. Set `REPO` to the destination for the code checkout.

```bash
set -eu
git clone --branch feature/logical-evidence https://github.com/SathyaHamsiniP/looking-beyond-attention.git "$REPO"
cd "$REPO"
GEN=three-operand-experiments/code/00-data-generation
python "$GEN/download_dataset.py" --output-dir "$WORK/historical"
python "$GEN/generate_three_operand_dataset.py" --renderer-root "$RENDERER" \
  --output-dir "$WORK/fresh" --num-images 32 --seed 41112
for runner in benchmark/code/{11-evidence-sufficiency,12-context-counterfactuals}/*.py; do
  name=$(basename "$runner" .py)
  python "$runner" prepare --historical "$WORK/historical" --data "$WORK/fresh" --out "$WORK/$name"
done
```

The renderer uses NumPy and Pillow. Use a fresh output folder. The preparation step checks the historical dataset fingerprint, rejects repeated pixels and verifies the object annotations. The entire new pool is evaluation-only, including images the generator internally tags as train. No tuning split is drawn from it. Both experiments can reuse these scenes. More images or a larger `--repeats` value can be used with a new output folder. The same truth configurations recur across runs.

Switch to a compatible CUDA runtime for inference. Install the [seven-rule dependencies](00-common/requirements-seven-rule-controls.txt). The existing loader downloads the pinned Qwen revision. Set the paths again after switching runtimes. For a short smoke run, use one of these commands first.

```bash
set -eu
cd "$REPO"
runner=benchmark/code/11-evidence-sufficiency/evidence_sufficiency.py
python "$runner" run --data "$WORK/fresh" --out "$WORK/evidence_sufficiency" \
  --model pretrained --mode visual --limit 2
```

Check the saved responses before committing more GPU time. To continue both experiments, run the following cell. You can run one model or modality at a time. Each completed response is saved atomically. Run the same command to resume. The runner checks the manifest, sources, software, images and adapter. Use one writer per condition. A full default comparison needs 1,008 responses for Experiment 11 and 2,576 for Experiment 12. Use `--rotations 1` in a separate folder prepared with the same manifest for a cheaper pilot. A one-order pilot cannot measure order sensitivity.

```bash
set -eu
cd "$REPO"
for runner in benchmark/code/{11-evidence-sufficiency,12-context-counterfactuals}/*.py; do
  out="$WORK/$(basename "$runner" .py)"
  for mode in visual symbolic; do
    python "$runner" run --data "$WORK/fresh" --out "$out" --model pretrained --mode "$mode"
    python "$runner" run --data "$WORK/fresh" --out "$out" --model lora --adapter "$ADAPTER" --mode "$mode"
  done
  python "$runner" report --out "$out"
done
```

Each condition writes `metrics.json`. The complete report writes `comparison.json` and `comparison.png`. Paired differences use chains or pairs as the unit. Scene variants can recur across those units, so the summaries are descriptive. Keep raw responses, manifests, scenes and weights in your working folder. The benchmark results tree holds compact design audits and can hold the finished comparison files after review.
