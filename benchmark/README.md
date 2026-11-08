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

## Experiment 01: diagnostic image only (new variation)

This belongs to **01 — Operator Controls**, not a new numbered experiment. Run `benchmark/diagnostic_only.py` after completing the original-compatible 10-vs-20 comparison. It reads its `paired_sets.json`, `config.json`, and 10-image `runs.jsonl`; deterministically picks the *first logically unique diagnostic image* from each original 10-image set and gives Qwen only that image. Candidate rules, model revision, original prompt wording (apart from image count), X/Y/Z answer rotations and probability readout stay fixed. It writes to a **separate** output directory and never modifies reference results.

```bash
python benchmark/diagnostic_only.py --reference-dir /path/to/original-operator-controls --data-dir /path/to/data-1000 --out /path/to/diagnostic-only --validate-only
python benchmark/diagnostic_only.py --reference-dir /path/to/original-operator-controls --data-dir /path/to/data-1000 --out /path/to/diagnostic-only --smoke-test
python benchmark/diagnostic_only.py --reference-dir /path/to/original-operator-controls --data-dir /path/to/data-1000 --out /path/to/diagnostic-only
```

Expected **24 matched diagnostic scenes × 3 label rotations = 72 trials**. Results: `config.json`, `selected_images.json`, resumable `runs.jsonl`, `summary.json` (includes 1-vs-10 paired accuracy change, error flips, cluster-bootstrap interval). The diagnostic image is unique *among R1/R2/R3*. Note that the original 10-image coverage sets may contain **more than one** scene with the diagnostic Boolean configuration, and the selection rule intentionally chooses one; this comparison changes both image count and amount of redundant evidence. It does not repeat the original remove-diagnostic/remove-control ablation.

## Experiment 02 — matched 1-vs-10 rule probes (provisional)

`probing.py` is a self-contained frozen-Qwen3.5-4B probe pipeline, independent of historical Experiment 02. It consumes the ten-image manifest from `build_sets.py`, and selects the **first uniquely diagnostic** scene from each ten-image set for the paired one-image condition. **This does not use the old 24-set operator-controls evaluation as its probe training data.** The new manifest must provide separate `train` and `test` partitions with image-disjoint source images, at least three sets per rule in both. Candidate rules, splits and counts are explicitly recorded, so Roshan's eventual benchmark choices require new rules/split/manifest artifacts, not rewriting extraction/scoring. Currently the runner supports exactly **three** candidate rules, matching the 1-vs-10 pilot; seven-rule tests require revising the model-answer mapping and identifiability checks later.

For a preliminary **illustrative** 30 train + 10 test sets per rule (not Roshan-approved sample sizes):

```bash
python benchmark/build_sets.py \
  --data-dir /path/to/data-1000 \
  --counts train=30 test=10 --images-per-set 10 \
  --out /path/to/probe-manifest-10.json

python benchmark/probing.py \
  --manifest /path/to/probe-manifest-10.json \
  --data-dir /path/to/data-1000 \
  --out /path/to/probing-1-vs-10 --validate-only
```

Then in Colab T4 with `transformers>=5.0`, `torch`, `bitsandbytes`, `accelerate`, `numpy`, `pillow`, `scikit-learn` installed:

```bash
python benchmark/probing.py \
  --manifest /path/to/probe-manifest-10.json \
  --data-dir /path/to/data-1000 \
  --out /path/to/probing-1-vs-10 --smoke-test

python benchmark/probing.py \
  --manifest /path/to/probe-manifest-10.json \
  --data-dir /path/to/data-1000 \
  --out /path/to/probing-1-vs-10
```

The smoke test extracts final-token activations at every decoder layer for one matched pair; it saves nothing. The full run saves per-set/per-condition compressed activation files, resumable `model_predictions.jsonl`, `layer_accuracy.csv`, `probe_predictions.csv`, and `summary.json`. It fits a standardized logistic-regression probe at **each layer** for each input size, with true and shuffled train labels, then reports accuracy on held-out image-disjoint sets. No model fine-tuning is performed. The runner uses answer rotation 0 for Qwen's final predictions; do not interpret those answer accuracies as label-rotation-averaged results. The last-token representation is an answer-adjacent probe target; decoding rule identity does not demonstrate that the model computes Boolean logic. The one-image case can be solved by recognizing one diagnostic ABC configuration.

The inference code requires a GPU smoke test before a full run; this new probing runner has not yet been executed against a Qwen model. Results are exploratory until the final aligned rules, partitions and fixed evaluation protocol are chosen. **Never merge draft results into the previously completed experiments.**
