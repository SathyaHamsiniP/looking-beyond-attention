# Common Boolean-rule benchmark — draft foundation

**Status: foundation only. No new experimental results.** This folder is self-contained and imports no code from Experiments 01–04. Only Python's standard library is required.

## Files

- `rules.json` — our original three-rule pilot truth tables; **not** the final seven-rule benchmark.
- `build_sets.py` — reads existing images and constructs a reproducible JSON list of image sets. It does **not** render images, fine-tune, run inference or alter earlier experiments.
- `test_build_sets.py` — small CPU-only tests covering ambiguity, image splits and nested comparisons.

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
