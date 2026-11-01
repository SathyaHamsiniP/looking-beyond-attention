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
this step as a **pilot**. In the completed Qwen3.5 run, early visual probes
reached 100% on the three held-out sets, according to the reported Colab results.
The diagnostic-scene and scene-count baselines also reached 100%. This does not
separate rule integration from recognizing a diagnostic scene pattern. The
underlying output files remain on Drive and were not independently inspected here.

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
tested here; the original Qwen3.5 pilot has since run in Colab, as reported above. Qwen3-VL
rule-probing results have not been reported here.

Activation patching remains future experiment 03. Reading out a rule does not
show that the model uses the representation in its answer. Later interventions
will address a different question, with their own controls and limitations.


## Combining evidence from more than one image

The initial Qwen3.5 pilot left us with an important limitation: a single positive
scene configuration could identify the rule. We now want to test a task where
neither of two diagnostic examples is enough on its own. This continues the
rule-probing investigation within Experiment 02; it is not a new numbered
experiment or an activation-patching study.

We use labelled positive and negative images. A positive example satisfies the
hidden rule; a negative example does not. These pairs of negative configurations
are jointly identifying:

| Hidden rule | First negative | Second negative | Candidates after first | Candidates after second |
|---|---|---|---|---|
| R1 | 010 | 100 | R1, R3 | R1, R2 |
| R2 | 001 | 100 | R2, R3 | R1, R2 |
| R3 | 001 | 010 | R2, R3 | R1, R3 |

Each column alone leaves two rules, while their intersection leaves only the
intended rule. Positive 111 and negative 000 are consistent with all three rules.
They are useful matched fillers, not additional identifying evidence.

`code/02-probing/rule_evidence.py` constructs and checks a four-slot task. The
paired condition contains the two diagnostic negatives, a positive 111, and a
negative 000. The first-only and second-only conditions replace the missing
diagnostic scene with a different 000 image. Every condition has four distinct
images, one positive label and three negative labels. The shared fillers and
retained diagnostic images are the same within a matched family. The labels
refer explicitly to Image 1, Image 2, and so on in the model prompt.

We cycle the four image positions and three answer-label assignments. Each image
appears equally often in each position, and each rule equally often under X, Y
and Z. These are four cyclic image orders, not all possible permutations. The
full comparison has 12 presentations per condition, three conditions per family,
and three held-out families: 108 forward passes. We reuse the existing Qwen3.5
4-bit loader and normalized next-token X/Y/Z probabilities, mapped back to rules.
There is no generated-reasoning evaluation or activation patching in this script.

### What this small pilot can tell us

We use variation blocks 0–1 and 2–3 for the baseline's training families, and
4–5 for testing. No image crosses the train/test boundary. Each block has one
family per rule. The six training families use 12 unique images; the three test
families use six unique images. Families within a block share filler and diagnostic
images, so the largest mutually image-disjoint collection is two training
families or one test family. The 108 presentations are not 108 independent tests.
The model is frozen; only the metadata scene-pattern baseline uses training data.

A symbolic baseline eliminates every rule inconsistent with the labelled scenes
and distributes probability uniformly over the survivors. A simple scene-pattern
baseline learns a table from labelled scene-count patterns to rule frequencies
in training, then applies it to the image-disjoint test families. It ignores
image order and visual appearance. For this exact design, both baselines assign
probability 1 to the true rule with both diagnostic examples and 1/2 with either
example alone. This is a checked property of the construction, not a measured
model finding. Neither baseline needs an abstract learned reasoning procedure.

We measure rule-selection accuracy, true-rule probability and probability mass
on logically consistent rules. Summaries average probabilities over presentation
orders and answer labels within each family before scoring. They also retain
mean presentation accuracy and paired changes from each single condition to the
paired condition. Exact ties receive fractional credit. The overall three-way
chance level is 1/3, but single-evidence conditions are intentionally ambiguous:
choosing the other consistent rule is not a logical error. A rational baseline
has only 1/2 target accuracy there. The task tests sensitivity to combined
evidence; even a paired improvement would not distinguish rule elimination from
memorized scene-pattern matching.

Four images require less sequence memory than the earlier ten-image prompts,
and the existing loaders can prepare these inputs. We provide a three-condition
smoke test before the 108-call pilot. This is a computational feasibility design,
not a claim that the new GPU run has passed. No model results for this evidence
comparison have been measured here.

For reliable evaluation, a useful next dataset would have at least 20 fresh,
image-disjoint held-out families per rule. Each matched family needs five unique
images across its three conditions, so 60 fully disjoint families require 300
held-out images, plus a separate training pool. Vary object arrangements,
backgrounds and distractors independently. More permutations of the current
images cannot supply that evidence. We have not changed or regenerated any of
the existing images.

### Running the evidence comparison

Mount Drive and select a fresh Colab GPU runtime. Use the new branch:

```bash
git clone --branch feature/evidence-integration https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
python -m pip install -r three-operand-experiments/code/requirements-qwen35.txt
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data'
export SCRIPT='three-operand-experiments/code/02-probing/rule_evidence.py'

# Logical, label, image-split and presentation-balance checks; no model or writes.
python "$SCRIPT" --data-dir "$DATA" --validate-only

# Save the validated design on Drive; no model inference.
python "$SCRIPT" --data-dir "$DATA" --generate-only

# Three matched conditions for one family, image order and answer layout.
python "$SCRIPT" --data-dir "$DATA" --smoke-test

# Complete pilot; rerun the same command to resume completed presentations.
python "$SCRIPT" --data-dir "$DATA"
```

If the checkout already exists, fetch and switch to `feature/evidence-integration`
instead of cloning over it. Use the actual existing `--data-dir` if your dataset
is elsewhere. Outputs default to the verified existing Drive results root,
under `probing/qwen3_5/rule/evidence_integration/`; `--experiment-dir` overrides
the root if Drive is mounted differently. The script never creates a missing
root. Smoke outputs are separate under `evidence_integration/smoke/`.

`config.json` contains the labelled examples, split, prompts, data hashes and
software versions. `predictions.jsonl` is the atomic, resumable checkpoint;
`predictions.csv` exposes all model and baseline probabilities.
`family_summary.csv`, `condition_summary.csv` and `paired_changes.csv` give the
matched comparisons. Interrupted runs reuse complete presentations and reject
incompatible configurations. All files stay on Drive; prior rule-probing
checkpoints and results are untouched. No result-upload or synchronization step
is required.
