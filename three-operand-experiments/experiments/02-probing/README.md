# 02 — Looking for concept information inside the models

After the [operator controls](../01-operator-controls/README.md), we were left
with a puzzle. The models could recognize the relevant objects and evaluate
explicit Boolean expressions, but struggled to identify a rule from examples.
We wanted to know whether useful information was present internally even when
the final answer did not use it successfully.

A probe is a small classifier trained on a frozen model's representations. Here
it predicts one of eight configurations, from A=0, B=0, C=0 through A=1, B=1,
C=1. A means green cube, B means blue sphere and C means red cylinder. This is
**concept probing**, not a test of whether the model inferred R1, R2 or R3.

## What we did

We reused the same 48 images: eight configurations with six visual variations
each. Variations 0–3 give 32 training images; variations 4–5 give 16 held-out test
images. Every class appears four times in training and twice in testing. No image
crosses the split. We do not regenerate images or change this split.

For each image we use the prompt “Describe the objects in this image.” Hooks
collect each decoder block's output in one forward pass, pooling immediately to
avoid retaining every token at every layer. Layer numbers start at 1 and exclude
the embedding input and final normalization. Qwen3.5 mixes attention and
linear-attention blocks; hooking block outputs lets us use the same comparison
without treating its internals as identical to Qwen3-VL.

We compare the mean of image-token representations, the mean of text-token
representations, and the last token. The image mean asks what is readable from
the image positions; text and last-token pooling ask what reaches other parts of
the prompt. These are pooling choices, not an exhaustive test of token positions.

At each layer and pooling choice, we fit a standard scaler and logistic
regression (C=1, seed 2026), fitting the scaler only on training images. We also
shuffle training labels and fit the same probe. This control helps check whether
the result depends on the correct concept labels. One shuffle is a sanity check,
not a statistical significance test. Eight-class chance accuracy is 12.5%.

## What we have observed so far

Both concept runs have finished on Colab. The researcher reports stronger best
visual-mean accuracy for Qwen3.5 and decodable visual concept information in both
models. **The actual CSV, JSON and PNG files are not available in this checkout,
so the numerical comparison is pending file verification.** We have not recreated
missing outputs or treated reported scores as independently verified results.
The copy command below prints each best observed layer accuracy directly from
the actual CSV; those values can then be added here with links to the files.

Even after verification, best observed layer accuracy is not an average over
layers. Choosing the best layer using this test set makes that score exploratory,
not an unbiased estimate for a separately selected layer. There are only 16 test
images; one changed prediction is 6.25 percentage points. These findings are
preliminary and do not establish successful Boolean rule induction or show which
model reasons better.

## Running in Colab

Mount Google Drive using Colab's Files panel and select a GPU runtime. Clone the
feature branch if needed:

```bash
git clone --branch feature/three-operand-probing https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
```

The intended Drive results root is
`/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results`.
The following commands use the sibling `data/` directory for the existing
`metadata.json` and `images/`. Check that it exists before running; if your images
are stored elsewhere, pass their existing directory to `--data-dir` instead.
No placeholder folders are created.

For Qwen3.5, in a fresh Colab runtime with CUDA PyTorch already installed:

```bash
cd /content/looking-beyond-attention
python -m pip install -r three-operand-experiments/code/requirements-qwen35.txt matplotlib
python three-operand-experiments/code/02-probing/probe_concepts.py \
  --model qwen3_5 --precision 4bit \
  --data-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data' \
  --experiment-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results' \
  --smoke-test
```

For Qwen3-VL use another fresh runtime, because the Transformers requirements
differ. Extract the supplied reference ZIP to
`/content/roshan/looking-beyond-attention-main` first:

```bash
cd /content/looking-beyond-attention
python -m pip install -r three-operand-experiments/code/requirements-qwen3vl.txt matplotlib
python three-operand-experiments/code/02-probing/probe_concepts.py \
  --model qwen3 --precision 4bit \
  --upstream '/content/roshan/looking-beyond-attention-main' \
  --data-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data' \
  --experiment-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results' \
  --smoke-test
```

Remove `--smoke-test` for a complete concept run. Rerun the same command to reuse
completed image checkpoints. `--validate-only` checks data and splits without
loading weights. The default loaders, quantization, prompts and calculations are
unchanged by the folder reorganization. The reorganized paths have CPU checks;
we have not rerun GPU extraction after the move.

`--experiment-dir` means the results root. The script appends
`probing/qwen3/concept/` or `probing/qwen3_5/concept/`; smoke runs use
`concept_smoke/`. Each full run keeps 48 pooled `.npz` checkpoints on Drive,
alongside `config.json`, `concept_predictions.csv` and `concept_accuracy.png`.
The CSV gives held-out predictions and true-class probabilities for each layer,
pool and control. The plot shows their accuracies across layers.

## Recovering and copying the completed runs

The earlier instructions accidentally led to a Drive folder literally named
`YOUR_EXISTING_EXPERIMENT_FOLDER`. First check where that folder exists. If it
is directly under MyDrive as in the earlier commands, run:

```bash
cd /content/looking-beyond-attention
python three-operand-experiments/code/02-probing/copy_concept_results.py \
  --recover-from '/content/drive/MyDrive/YOUR_EXISTING_EXPERIMENT_FOLDER'
```

That argument refers to the **existing mistaken folder**, not a new output path.
If it is elsewhere, use its actual location. The script copies the completed
concept outputs, including activation checkpoints, to the intended Drive root.
It checks file hashes and refuses conflicting destinations. It never deletes the
originals. Keep the originals until both models have passed the printed checks.

The same command then copies only the three lightweight files per model into
`three-operand-experiments/results/02-probing/qwen3/concept/` and
`qwen3_5/concept/` in the Colab checkout. It checks model/configuration metadata,
the 16 held-out image labels, layer/pool/control coverage, CSV correctness, and
PNG integrity. It reads existing results; it never fits probes or creates plots.
For files already in the correct Drive location, omit `--recover-from`.
Use `--drive-results` only if you intentionally use another results root.

After checking the printed best-layer scores and files, review and commit those
six lightweight outputs on the feature branch. The activation checkpoints must
stay on Drive; `.gitignore` excludes `.npz` and model checkpoints. The copy checks
validate output structure and consistency, not the original GPU execution or
whether a plot exactly matches its CSV.

## What we want to test next

The next step is rule probing after ten images. It will need grouped scene-set
splits, answer-layout controls, shuffled labels and an eight-bin scene-count
baseline. Recovering a rule from image frequencies alone would not show abstract
reasoning. With only two held-out variations per configuration, each rule has
just ten positive held-out images; repeated prompts would not be independent
new evidence. We need to account for this before interpreting rule probes.

Later, experiment 03 will use activation patching to ask whether replacing a
representation changes rule probabilities. Self-patches and same-rule donors
will be controls for different-rule donors. Neither rule probing nor patching is
implemented yet. Decoding information does not show that a model uses it, and a
patch can introduce an unnatural internal state. These are separate questions.
