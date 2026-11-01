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

Both concept runs have finished on Colab. On the 16 held-out images, the
researcher reported these best observed layer accuracies:

| Representation | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Visual-token mean | 68.75% | 87.5% |
| Text-token mean | 62.5% | 43.75% |
| Last token | 43.75% | 43.75% |

These are measured scores reported from the completed runs, not synthetic
examples or averages across layers. The underlying files remain on Drive and
have not been independently inspected in this checkout. Both models contain
decodable visual concept information, but this does not establish successful
Boolean rule induction.

Best observed layer accuracy is not an average over
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

## Keeping outputs on Drive

All outputs stay under the verified Drive results root. There is no result-copy,
synchronization or Git commit step. Read the CSVs and plots on Drive, then update
this README with concise measured findings. Existing tracked results remain as
an archive. The results root must already exist; a missing root produces an error
asking for the actual path via `--experiment-dir`, rather than creating a new
folder. Only this run's experiment subfolders are created.

If the earlier runs are still in the mistakenly named Drive folder, copy them
within Drive to the intended project results folder using Drive's file manager.
Check both copies before deleting anything. No automatic migration is performed.

## Next: can we read out the rule after ten images?

The concept results tell us that image information is readable. We now want to
ask whether a linear classifier can recover R1, R2 or R3 after the model has
processed a complete ten-image set. `code/02-probing/probe_rules.py` implements
this next step as a **pilot**. No GPU rule-probing results have been measured yet.

We reuse the operator-control sampler and both model loaders. We also reuse the
concept probe's extraction function, collecting visual-token mean, text-token
mean and last-token representations at every decoder block. The neutral prompt
names the three concepts and asks for the Boolean relationship explaining the
ten positive images. It gives no generating rule, candidate expressions, R1/R2/R3
labels or answer ordering. We do not mix the old multiple-choice prompt into
this condition. This is a readout from the input pass, before any generated answer.

### Why this is only a pilot

A random split of sampled sets would leak the same images into training and
testing. Instead, training uses variations 0–3 and testing uses variations 4–5,
with no shared images across the two partitions. The script rejects duplicate
unordered sets and uses just one presentation of each set. It restores the
original sampler pool after constructing the split.

| Partition | Unique images actually used | Distinct ten-image sets | Sets per rule |
|---|---:|---:|---:|
| Training | 28 | 24 | 8 |
| Testing | 14 | 3 | 1 |

Six 000 images are unused because they are negative for all three rules. All
training sets reuse images within training. Each rule has five positive
configurations; with two held-out variations of each, there are exactly ten
positive test images for that rule. Any coverage-based ten-image test set must
therefore use all ten. Reordering these images would not create another
underlying set. Test sets for different rules also overlap with one another.
For this fixed sample, the largest collection of mutually image-disjoint sets
within either partition contains only one set. The configuration reports these
counts rather than calling 27 presentations independent observations.

The test has only three predictions per layer and pooling choice. Accuracy moves
in steps of 33.3 percentage points; even 100% would mean only three correct
predictions on overlapping sets. We do not compute independent-set confidence
intervals or select and advertise a best layer as a generalization result.

For a stronger evaluation we need fresh rendered scenes, not extra permutations
of these images. As a practical starting point, ten mutually image-disjoint
held-out sets per rule would require 300 held-out image instances with distinct
underlying scenes, separate from training data. Training also needs a larger,
separate scene pool. More independently generated backgrounds, distractors and
object arrangements would help test visual generalization. We have not generated
or requested changes to the existing dataset for this pilot.

### Controls and saved outputs

Each layer/pool uses the same training and testing sets, a training-only standard
scaler, and logistic regression with C=1 and seed 2026. A shuffled-training-label
probe uses the same features and split. This is one fixed shuffle, not a
significance test. The plot includes the 1/3 chance line.

We fit the same classifier to an eight-bin vector of ground-truth scene
frequencies, with its own shuffled-label control. We also include a direct
metadata baseline: 001 identifies R1, 010 identifies R2, and 100 identifies R3.
Coverage construction always includes the relevant diagnostic type. That baseline
can identify the rule by construction, without inspecting model representations.
It is an oracle based on scene annotations, not model perception accuracy. Its
perfect performance would be a property of the task design, not evidence of a
successful model experiment. A hidden-state probe could simply detect that type;
this pilot cannot distinguish that shortcut from abstract rule induction.

All outputs stay on Drive under `probing/qwen3/rule/` or
`probing/qwen3_5/rule/`: `config.json` records the prompt, split manifest, data
hashes and package versions; each set has a pooled `.npz` checkpoint;
`predictions.csv` gives test predictions and R1/R2/R3 probabilities;
`layer_accuracy.csv` gives probe and baseline accuracies; and `rule_accuracy.png`
plots the layer comparison. Checkpoints are written through temporary files and
renamed when complete. Resume refuses different settings, data or package
versions instead of silently combining runs. Keep the same reference wrapper
and runtime setup when resuming Qwen3-VL.

### Running the rule pilot in Colab

Use a fresh GPU runtime for each model, mount Drive, and work from the
`feature/rule-probing` branch. If you already have a checkout, fetch and switch to
that branch instead of cloning over it. From a fresh runtime:

```bash
git clone --branch feature/rule-probing https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
```

The commands below use the existing sibling `data/` folder on Drive. If your
images live elsewhere, pass their actual existing directory with `--data-dir`.
The results root defaults to the existing project results folder. It is checked
before any files are saved; if Drive is mounted differently, supply the actual
existing root with `--experiment-dir`.

Qwen3.5 (CUDA PyTorch is already supplied by Colab):

```bash
python -m pip install -r three-operand-experiments/code/requirements-qwen35.txt matplotlib
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data'
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3_5 --data-dir "$DATA" --validate-only
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3_5 --data-dir "$DATA" --precision 4bit --smoke-test
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3_5 --data-dir "$DATA" --precision 4bit
```

Qwen3-VL, in another runtime, after extracting the supplied reference repository
to `/content/roshan/looking-beyond-attention-main`:

```bash
python -m pip install -r three-operand-experiments/code/requirements-qwen3vl.txt matplotlib
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data'
export UPSTREAM='/content/roshan/looking-beyond-attention-main'
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3 --data-dir "$DATA" --validate-only
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3 --data-dir "$DATA" --upstream "$UPSTREAM" --precision 4bit --smoke-test
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3 --data-dir "$DATA" --upstream "$UPSTREAM" --precision 4bit
```

Validation is CPU-only and writes nothing. The smoke test extracts two complete
ten-image sets (one training R1 set and one held-out R2 set) into `rule/smoke/`;
it never fits probes or reports accuracy. The last command runs the full
27-set pilot, then fits the probes and baselines. Rerun it after a disconnection
to reuse completed checkpoints. Ten-image extraction uses more GPU memory than
single-image concept extraction; check the smoke test before starting the pilot.
The folder structure, CPU split checks and synthetic software checks have been
tested here; GPU extraction must still be checked in Colab.

[Experiment 03](../03-activation-patching/README.md) now implements final-token
activation patching; its GPU findings remain pending. Reading out a rule does
not show that the model uses that representation in its answer. These
interventions address a different question, with their own controls and limitations.
