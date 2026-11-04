# Implementation delivery

Branch: `feature/complete-research-pipeline`. Starting commit: `4fd6e0c12db368c3aada60cc0efb98a337bcae69`. No changes or merges to main.

## Audit and implementation checklist

- Audited the existing 01–03 runners, samplers, model adapter, three-pool probing, result formats, archived pilot results, dependency files and tests.
- Kept the legacy baseline patching runner, imports and output manager unchanged. Its active Drive path is protected by using entirely separate new entry points; Drive was not accessed.
- Added deterministic 800/100/100 derived splits, V1/V2/V3 generation, singleton/consistency checks and symbolic oracle validation. Source images and metadata remain untouched.
- Added variable-option/full-sequence answer scoring, manifest validation, runtime token checks, revision/dependency provenance, input hashes, content-addressed directories, and duplicate-checked journals.
- Added manifest evaluation, explicit concept/rule partitions and layer-wise probes.
- Added runtime-discovered hybrid-layer QLoRA targets, across-layer and fresh single-layer entry points, frozen-base checks, token-aligned targets, validation selection, trusted checkpoint resumption and adapter compatibility metadata.
- Added separate matched baseline/adapter patching and complete-plan comparisons. Receiver-probability outcomes are defined identically across same/different-rule conditions.
- Added documentation and CPU tests. No pretrained model was downloaded or run. GPU validation remains outstanding.

## Final source tree

Generated caches, the virtual environment, large data and existing archive contents are omitted from this source view. Historical directories remain in place.

```text
README.md
three-operand-experiments/
├── COLAB.md
├── IMPLEMENTATION.md
├── tasks/
│   ├── __init__.py
│   ├── README.md
│   ├── rules.py
│   ├── build_sets.py
│   ├── validate_sets.py
│   ├── configs/
│   │   ├── v1-positive-3rules.json
│   │   ├── v2-mixed-3rules.json
│   │   └── v3-mixed-varied-rules.json
│   └── splits/dataset-v1-seed2026.json
├── code/
│   ├── 00-data-generation/                  [preserved]
│   ├── 01-operator-controls/
│   │   ├── operator_controls_qwen3.py      [preserved]
│   │   ├── operator_controls_qwen35.py     [preserved]
│   │   └── evaluate_tasks.py               [new]
│   ├── 02-probing/
│   │   ├── probe_concepts.py               [preserved]
│   │   ├── probe_rules.py                  [preserved]
│   │   └── probe_tasks.py                  [new]
│   ├── 03-activation-patching/
│   │   ├── patch_activations_qwen35.py     [preserved]
│   │   ├── test_patching.py                [one bracket fixed]
│   │   └── patch_tasks.py                  [new]
│   ├── 04-lora/
│   │   ├── train_lora.py
│   │   ├── train_single_layer.py
│   │   └── evaluate_lora.py
│   ├── 05-post-lora-comparison/
│   │   ├── patch_finetuned.py
│   │   └── compare_results.py
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── model_qwen35.py
│   │   ├── data_loader.py
│   │   ├── results_paths.py
│   │   ├── run_utils.py
│   │   ├── evaluation.py
│   │   ├── patching.py
│   │   └── comparison.py
│   ├── results_paths.py                    [preserved legacy manager]
│   ├── requirements-qwen35.txt             [PEFT/matplotlib added]
│   └── requirements-qwen3vl.txt            [preserved]
├── experiments/
│   ├── 01-operator-controls/README.md      [extended]
│   ├── 02-probing/README.md                [extended]
│   ├── 03-activation-patching/README.md    [status clarified/extended]
│   ├── 04-lora/README.md                   [new]
│   └── 05-post-lora-comparison/README.md   [new]
├── data/                                  [preserved pilot images/metadata]
├── results/                               [preserved historical archives]
└── tests/
    ├── test_operator_controls_qwen3.py     [preserved]
    ├── test_operator_controls_qwen35.py    [preserved]
    ├── test_probe_rules.py                 [preserved]
    ├── test_results_paths.py               [preserved]
    └── test_pipeline.py                    [new]
```

## Validation performed

- `python -m unittest discover -s three-operand-experiments/tests -v`: **26 passed**. Tests include deterministic generation, leakage/tamper rejection, all truth-table uniqueness, options/labels, full multi-token likelihoods, patch-hook cleanup, real PEFT on tiny randomly initialized Qwen3.5 hybrid models, frozen weights, selected targets, adapter reload, full training/checkpoint restoration, probe partition accounting, result resume/collisions, and matched comparison completion checks.
- `python three-operand-experiments/code/03-activation-patching/test_patching.py`: **2 passed** after correcting the pre-existing bracket syntax error.
- Original `patch_activations_qwen35.py --validate-only`: **passed**, 18 pairs (6 same / 12 different), held-out-only.
- Actual downloaded dataset-v1: original downloader verified its published ZIP checksum and 1,000 images (800/200). No regeneration or replacement occurred.
- Generated/validated V1 **108 sets**, V2 **108 sets**, V3 **288 sets**, each with 800/100/100 image split and unique symbolic oracle answers. No individual labelled V2/V3 observation uniquely identifies its rule.
- All new model entry points passed `--validate-only`, including concept/rule probes, both training modes, evaluation and baseline/post-LoRA patching. V3 default patching plan has **64 image-disjoint pairs**.
- `uv pip check`: compatible installed packages. Tested versions include Python 3.12.14, torch 2.15.0+cpu, Transformers 5.22.1, PEFT 0.21.2, bitsandbytes 0.50.2. Optional kernel fallbacks were observed; optimized kernels were not made mandatory.
- New Python source formatted. Targeted Ruff correctness checks, compilation and `git diff --check` passed.

CPU test fixtures may use synthetic records or random model weights. They are software tests, not research results. Synthetic comparison plots generated by tests were temporary and were not committed.

## Commands and output locations

[COLAB.md](COLAB.md) contains complete executable commands for dataset download, V2/V3 validation, historical and new Experiment 01, both concept/rule probe protocols, the untuned V3 baseline, across-layer and single-layer QLoRA, validation/final-test evaluation, baseline and post-LoRA patching, smoke tests, and final comparisons. Its output table gives exact Drive path templates, including the protected historical location. Every new runner prints its actual configuration-derived run directory.

## Remaining limitations and external access

- Real-image Qwen3.5 tokenizer mappings are verified at runtime but were not downloaded/tested here. Complete multi-token scoring is CPU-tested with controlled tokenizers.
- Pretrained 4-bit GPU inference/training, T4 memory limits with ten images, GPU checkpoint recovery and new post-LoRA patching require Colab smoke validation. Do not call those workflows GPU-validated yet. Use a pinned model revision for research runs.
- No new pretrained-model metrics, LoRA outcomes or patching research findings exist. The user reports a passed historical Colab patching smoke test and an active full baseline; its Drive files were not inspected.
- New mixed-task manifests are image-disjoint across partitions but share logical configurations and rules. Within-partition overlapping sets are explicitly marked and are not independent samples. Cyclic option controls do not cover all 8! permutations. Label balance does not eliminate every configuration-frequency shortcut.
- Legacy V1 patching output cannot be treated as matched V3 evidence. Comparison accepts explicit paths, reads sources without changing them, and rejects incompatible protocols. Run the new manifest baseline for a valid before/after comparison.
- Google Drive and GitHub connector tools are unavailable in this session. HTTPS Git reads work. The GitHub API operation needed for a draft PR returned `Forbidden`; no draft PR is claimed unless separately confirmed.
