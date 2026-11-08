# Common Boolean-rule benchmark — draft foundation

**Status: foundation only. No new experimental results.** This folder is self-contained and imports no code from Experiments 01–04. The dataset builder and CPU tests use only Python's standard library. GPU inference requires PyTorch, Transformers, bitsandbytes, accelerate and Pillow.

## Files

- `rules.json` — our original three-rule pilot truth tables; **not** the final seven-rule benchmark.
- `build_sets.py` — reads existing images and constructs a reproducible JSON list of image sets. It does **not** render images, fine-tune, run inference or alter earlier experiments.
- `test_build_sets.py` — small CPU-only tests covering ambiguity, image splits and nested comparisons.
- `pilot_10_vs_20.py` — standalone pretrained Qwen3.5-4B paired operator-controls pilot (no imports from older experiments).

A **set** is one rule-identification question containing several images. The builder's `--counts` are sets **per rule**, not individual images.

## What the builder guarantees

- Every generated set contains each satisfying Boolean configuration at least once and has no repeated image.
- **Uniqueness among supplied candidate rules** is checked. For positive-only tasks, a rule is compatible with a set if it satisfies every image in that set. If more than one candidate is compatible, generation stops with an error.
- Images are disjoint between train/validation/test partitions. Images may appear in multiple sets **within** the same partition, and that reuse must be reported in any evaluation.
- With the same seed, a 10-image set is the prefix of its corresponding 20-image set, and a smaller training budget is a subset of a larger budget. Image set identity does not depend on image count.
- Rule truth tables and splits are **data**, not hidden in copied experiment code.

**Caution:** The existing rules are identifiable among R1/R2/R3 but could be satisfied by other Boolean formulas. A single decisive scene can reveal the rule; this is not yet a proof of reasoning across all images.

## What remains on hold until Roshan replies

- His latest seven operators and whether we should use a positive-only or positive/negative benchmark.
- Final number of training/validation/test **sets** and split construction. **300 training sets is not a fixed target.**
- Shared 2-operand vs 3-operand rule families, 3-vs-7 comparison, prompts, Qwen settings, option rotations and evaluation metrics.
- Any new GPU training, probing or activation-patching runs.

## Quick checks

From the `benchmark/` directory:

```bash
python -m unittest discover -p 'test_*.py'
python build_sets.py --help
```

To create **only a draft manifest**, supply a folder with `metadata.json` (records with `filename` and Boolean `bits`) and `images/`. The `--partitions` option accepts either a filename-to-partition JSON mapping or an object containing `image_partitions`. If omitted, the builder uses each record's existing `split`; it never invents splits.

Example with **illustrative, not agreed** counts (from the repository root):

```bash
python benchmark/build_sets.py \
  --data-dir /path/to/data-1000 \
  --partitions /path/to/agreed_partitions.json \
  --counts train=2 validation=1 test=1 \
  --images-per-set 10 \
  --out /path/to/draft_manifest.json
```

Output files are never silently overwritten if their content differs. Do not treat this preliminary three-rule generator as the finalized benchmark or use its drafts to claim matched results with Roshan. When the protocol is agreed, use the same manifest for operator controls, probing, activation patching, and LoRA rather than creating four separate datasets.

## Running the 10 vs 20 pilot on Colab T4

Use the generated `pilot-10.json` and `pilot-20.json` manifests and the same `data-1000` folder. From the repo root:

```bash
python benchmark/pilot_10_vs_20.py --manifest-10 /path/to/pilot-10.json --manifest-20 /path/to/pilot-20.json --data-dir /path/to/data-1000 --out /path/to/results --validate-only
pip install 'transformers>=5.0' accelerate bitsandbytes pillow
python benchmark/pilot_10_vs_20.py --manifest-10 /path/to/pilot-10.json --manifest-20 /path/to/pilot-20.json --data-dir /path/to/data-1000 --out /path/to/results --smoke-test
python benchmark/pilot_10_vs_20.py --manifest-10 /path/to/pilot-10.json --manifest-20 /path/to/pilot-20.json --data-dir /path/to/data-1000 --out /path/to/results
```

The smoke test runs one 20-image forward pass without saving results. The full pilot runs 6 matched sets × 3 answer rotations × 2 sizes = 36 evaluations, saving `config.json`, incremental `predictions.jsonl`, and `summary.json` on Drive. It uses frozen pinned-revision Qwen3.5-4B with 4-bit NF4, not a fine-tuned adapter. This pilot uses the existing source `test` partition for diagnostics: **not an untouched final benchmark test**. Keep future final test examples locked. No LoRA or patching is run.

## Original-compatible operator controls (recommended; replaces the earlier exploratory pilot)

Run `operator_controls.py` rather than `pilot_10_vs_20.py` for the research comparison. It reproduces the original `main_72` setup with **8 sets/rule, 3 rules, and 3 cyclic label rotations**; the *only intended intervention* is changing the number of images per set, from 10 to 20. It recreates the original 10-image selections from the same `data-1000` training image pool, seed 2026, and appends 10 additional satisfying images. The original 10-image order is preserved. It uses the original prompt wording, 4-bit NF4 Qwen3.5-4B, and restricted next-token X/Y/Z softmax. The original model loader did not pin a specific revision; the new runner pins the revision previously recorded for Experiment 04 to make future runs reproducible. Compare results with original runs only after confirming the exact model revision and software environment.

```bash
python benchmark/operator_controls.py --data-dir /path/to/data-1000 --out /path/to/operator-controls-10-vs-20 --validate-only
python benchmark/operator_controls.py --data-dir /path/to/data-1000 --out /path/to/operator-controls-10-vs-20 --smoke-test
python benchmark/operator_controls.py --data-dir /path/to/data-1000 --out /path/to/operator-controls-10-vs-20
```

The full run uses **24 paired image sets × 3 option rotations × 2 image counts = 144 inferences**. It saves `config.json`, `paired_sets.json`, resumable `runs.jsonl`, and `summary.json`. It has no training, no validation-based model selection, and no need for train/test splits for the operator-controls task; it uses the original historical `train` image pool solely to reproduce that experiment's sampling. The same rule-identifiable sets are reused across rotations, so 144 is not the number of statistically independent examples. The new runner has not yet been executed on a GPU and must pass its smoke test first.
