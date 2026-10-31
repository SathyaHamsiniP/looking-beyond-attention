# Looking Beyond Attention

A vision-language model might recognize everything in an image and still struggle to reason about what it sees. This project explores that gap.

We're interested in understanding whether vision-language models actually use the information available to them when solving logical problems, or whether their predictions are influenced by simpler patterns and biases.

Rather than relying only on attention weights to understand model behavior, we use controlled experiments to test what information a model recognizes, how it combines evidence, and how its answers change when that evidence is modified.

## What we're working on

Our first set of experiments focuses on **three-operand Boolean rule identification**.

We generate CLEVR-style images containing combinations of three concepts:

- **A:** Green cube
- **B:** Blue sphere
- **C:** Red cylinder

Each set of images is generated using one of three Boolean rules. The model sees ten examples and must identify which rule explains them.

We tested this with **Qwen3-VL-4B-Instruct** and **Qwen3.5-4B**.

To understand what might be going wrong, we also tested whether the models could recognize the objects, evaluate Boolean operations, use decisive examples, and remain consistent when answer choices were rearranged.

[Read more about the experiments](three-operand-experiments/experiments/04-three-operand-operator-controls/README.md)

## What we found

| Task | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Rule identification | 33.3% | 31.9% |
| Object recognition (8 images) | 100% | 100% |
| Boolean gate evaluation (both stages correct) | 79.2% | 95.8% |

Both models recognized the objects correctly in our small perception test and performed considerably better when asked to evaluate explicit Boolean operations.

However, neither reliably identified the rule behind a set of images. Their accuracy remained close to the three-choice chance level of 33.3%.

We also observed strong answer-position biases. For example, Qwen3-VL selected the first displayed option in 143 out of 144 trials in our option-order experiment.

These findings suggest that **recognizing concepts and evaluating individual logical expressions do not necessarily translate into inferring a rule from multiple examples**.

The experiments are small and controlled, so we don't treat these observations as a general verdict on the models' reasoning abilities.

## Repository structure

- `three-operand-experiments/code/` — Python implementations for both models and their dependencies
- `three-operand-experiments/data/` — Generated images and metadata
- `three-operand-experiments/tests/` — Automated checks for the experimental code
- `three-operand-experiments/results/` — Recorded predictions and experiment summaries
- `three-operand-experiments/experiments/` — Experimental methodology and findings

The experiments are implemented as Python scripts and can be run on a machine with suitable GPU resources. Each model has its own requirements file.

The Qwen3.5 Python runner has been tested with actual model inference. The converted Qwen3-VL runner has passed CPU-based tests but still needs a separate GPU verification.

## Project status

This is ongoing research. We're using these initial experiments to understand the difference between perceiving evidence, carrying out logical operations, and identifying relationships across multiple examples.

## From behaviour to internal representations

Roshan's earlier two-operand experiments are our methodological starting point.
Our extension asks whether the same questions become harder with three concepts
and the rules R1 = (A AND B) OR C, R2 = (A AND C) OR B, and
R3 = (B AND C) OR A. The uploaded reference is separate from the maintained
three-operand experiments; we do not replace our dataset with its two-operand data.

Our previous runs found weak rule identification despite good object recognition
and explicit Boolean evaluation. Removing decisive images and removing other
images produced little change in output probabilities in the earlier controls;
this is especially documented for Qwen3-VL. These observations motivate a new
question: could the internal representations contain useful information that the
final answer does not reveal? They do not establish that this is what happens.

A probe is a small classifier trained on frozen model representations. First we
ask it to distinguish the eight A/B/C configurations from one image. Later we
will ask it to identify the generating rule after the model sees ten images.
Following Roshan's probing approach, we collect decoder-layer outputs at the last
token, averaged over text tokens, and averaged over image tokens. These are three
pooling choices, not an exhaustive search over token positions. Layer numbers
start at 1 and refer to block outputs, excluding embeddings and final normalization.
Qwen3.5 has mixed decoder block types, so extraction hooks their outputs rather
than assuming every block contains the same attention module as Qwen3-VL.

## First probing stage

Only `three-operand-experiments/code/probe_three_operand.py` is new at this stage.
It reuses the existing loaders and checks the existing 48-image dataset. No images
are generated. The concept probe trains on variations 0–3 and tests on variations
4–5: 32 training images and 16 test images, covering all eight classes in both.
Scaling is fitted on training data only. Logistic regression uses fixed C=1;
there is no tuning against the test set. A shuffled-training-label control uses
the same split. Chance accuracy is 1/8.

Each image's pooled features are saved separately in a compressed NumPy file;
this is a numerical checkpoint, not a model result table. Completed images are
reused after a Colab restart. Configuration and dataset hashes prevent mixing
incompatible checkpoints. Resume with the same model software environment.
`concept_predictions.csv` records every held-out prediction and true-class
probability by layer, pool and control. `concept_accuracy.png` plots accuracy.
The extraction smoke test uses two images and never reports probe accuracy.

## Running the first stage in Colab

Select a GPU runtime. Mount Drive through Colab's Files panel. Keep the checkout
in `/content/looking-beyond-attention` and extract Roshan's supplied ZIP separately,
for example to `/content/roshan/looking-beyond-attention-main`. The external
wrapper is needed only for Qwen3-VL. Use a fresh runtime for each model because
the requirements specify different Transformers versions.

Set these paths to your **existing** Drive directories. The dataset directory
must contain `metadata.json` and `images/`. There was no fixed Drive path in the
maintained runners, so these examples deliberately do not guess your folder name.
Run the following in a Colab terminal (or a shell cell):

```bash
cd /content/looking-beyond-attention
export DATA_DIR="/content/drive/MyDrive/YOUR_EXISTING_DATA_FOLDER"
export EXPERIMENT_DIR="/content/drive/MyDrive/YOUR_EXISTING_EXPERIMENT_FOLDER"

# Qwen3.5: Colab already supplies CUDA PyTorch.
python -m pip install -r three-operand-experiments/code/requirements-qwen35.txt matplotlib
python three-operand-experiments/code/probe_three_operand.py \
  --model qwen3_5 --data-dir "$DATA_DIR" --experiment-dir "$EXPERIMENT_DIR" --validate-only
python three-operand-experiments/code/probe_three_operand.py \
  --model qwen3_5 --data-dir "$DATA_DIR" --experiment-dir "$EXPERIMENT_DIR" --smoke-test
# After checking the smoke output, remove --smoke-test to extract all images and fit probes.
python three-operand-experiments/code/probe_three_operand.py \
  --model qwen3_5 --data-dir "$DATA_DIR" --experiment-dir "$EXPERIMENT_DIR"
```

For Qwen3-VL, in a separate fresh GPU runtime:

```bash
cd /content/looking-beyond-attention
export DATA_DIR="/content/drive/MyDrive/YOUR_EXISTING_DATA_FOLDER"
export EXPERIMENT_DIR="/content/drive/MyDrive/YOUR_EXISTING_EXPERIMENT_FOLDER"
python -m pip install -r three-operand-experiments/code/requirements-qwen3vl.txt matplotlib
python three-operand-experiments/code/probe_three_operand.py \
  --model qwen3 --upstream /content/roshan/looking-beyond-attention-main \
  --data-dir "$DATA_DIR" --experiment-dir "$EXPERIMENT_DIR" --smoke-test
# Run the same command without --smoke-test for the complete concept probe.
```

The default precision is NF4 4-bit, matching the previous loaders. Outputs go to
`$EXPERIMENT_DIR/probing/qwen3/concept/` or `probing/qwen3_5/concept/`;
smoke outputs use `concept_smoke/`. Existing experiment outputs are untouched.
Rerun the same command after disconnection to resume extraction.

## What comes next

Rule probing and activation patching are **pending**, rather than implemented
experiments. Rule probes will use the existing ten-image prompts, evaluate all
trials (with correctness reported separately), and keep answer-label variants of
the same set together. We will separate image variations across train and test,
avoid duplicate scene sets across splits, and report an eight-bin scene-frequency
baseline alongside hidden-state probes. The current image pool is small: coverage
sampling needs ten images, while held-out variations provide only ten positive
images per rule. We must account for this constraint before claiming independent
rule-probe test samples. More sampled prompts do not create more independent images.

Layer and pooling comparisons will use the same splits. Shuffled labels,
text-only inputs and answer-layout controls will help check shortcuts. Scene
counts may already determine the rule in these controlled sets; successful rule
classification alone would not show abstract rule reasoning, even if a hidden-state
probe exceeds that baseline.

Patching will then replace selected decoder representations in a recipient prompt
with those from a donor and compare X/Y/Z probabilities mapped back to R1/R2/R3.
Self-patches, same-rule donors and different-rule donors will provide controls.
Token alignment and Qwen3.5's mixed block architecture need checking before this
is implemented. Results will belong under `activation_patching/qwen3/` and
`activation_patching/qwen3_5/` in the same configurable experiment folder.

## New findings: pending GPU runs

No concept-probe accuracy or activation-patching result has been measured yet.
CPU data checks and synthetic software checks do not establish that GPU extraction
works. Record the Colab software versions and inspect the two-image extraction
before launching the full concept probe.

With only six visual variations per concept configuration, any measured accuracy
will have substantial uncertainty and limited visual generalization. Reading
information with a classifier does not show the model uses it. A patch-induced
change would provide intervention evidence, but could also reflect an unnatural
activation replacement. Neither experiment alone proves a model implements
Boolean reasoning.
