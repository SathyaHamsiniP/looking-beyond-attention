# 03 — Do internal representations change the rule prediction?

Our earlier experiments separated recognizing concepts from identifying a rule.
The concept probes found readable image information, but that did not show that
the model used it to choose a rule. In the reported Qwen3.5 rule-probing pilot,
early visual probes reached 100% on three held-out sets, but scene-count and
diagnostic-scene baselines did too. We therefore cannot take a readable rule
label as evidence that the model is carrying out Boolean reasoning.

Activation patching asks a different question. We run one input, save an internal
representation, and insert it into another input's computation. We then measure
whether the second input's answer changes. This is a controlled intervention on
an activation, not a test of general reasoning ability. This implementation starts
with Qwen3.5-4B only. **No Qwen3.5 GPU patching results have been measured here.**

## Keeping the image pairs closely matched

We reuse the existing 48 images and the same rules:

| Rule | Expression | Identifying positive configuration |
|---|---|---|
| R1 | (A AND B) OR C | 001 |
| R2 | (A AND C) OR B | 010 |
| R3 | (B AND C) OR A | 100 |

A is a green cube, B a blue sphere, and C a red cylinder. For each of R1/R2,
R1/R3 and R2/R3, we select nine distinct images satisfying both rules. We cover
all four shared positive configurations before sampling the remaining shared
images. The tenth image is the relevant one-hot diagnostic configuration.
Changing this one image switches the uniquely consistent rule. Every input has
ten distinct images, and the two inputs have identical images in nine slots.
CPU validation checks uniqueness against all three candidate rules, not just
the two being compared.

We begin with one matched pair per rule pair: three pairs using 26 unique images,
including the same-rule control images. The diagnostic occupies slot 1, 4 or 7
across the three default pairs; its slot and the other image positions are matched
within each pair. This is not a fully position-balanced design. The same-rule
donor changes the diagnostic image to another visual variation of the same
configuration, leaving all nine other images unchanged.

Pairs reuse images. Both directions, label rotations and layers are repeated
measurements of these inputs, not extra independent examples. There is no probe
training or train/test split in this intervention experiment. More sampled pairs
from the same 48 images would not provide broad visual generalization.

## What we patch and measure

We reuse the existing loader, its NF4 4-bit settings, the original ten-image
multiple-choice prompt and X/Y/Z next-token readout. Donor and recipient always
have exactly the same prompt and rule-to-label mapping within a comparison.
Three cyclic assignments rotate each rule through X, Y and Z. We also reverse
which of the two inputs is the donor.

The donor is run without a patch. A forward hook at each decoder block saves
only its final-token output, before the final decoder normalization. We run the
recipient unpatched too, saving its answer logits and the same per-layer states.
For a patch, we start a fresh recipient forward pass and replace its final-token
state at **one** block with that block's saved donor state. Subsequent blocks run
normally on the recipient sequence. No other token is replaced.

The inputs may tokenize to different lengths. We use the last position in each
sequence rather than requiring an index-by-index token alignment. The block
hooks support the mixed decoder blocks in Qwen3.5 and preserve tuple outputs
where needed. Saved states are CPU copies. Replacement checks hidden width,
floating dtype and finite values, then converts to the recipient block's device
and dtype. The hook returns a cloned output; neither the saved state nor the
original output is changed in place. Every forward uses inference mode and
`use_cache=False`, and hooks are removed even if a forward fails.

For each run we save the restricted candidate logits and normalized probabilities.
The main effect is the change in:

`logit(donor rule) - logit(recipient rule)`

A positive change means the patch shifted this comparison toward the donor rule.
It does not necessarily mean the model became correct, that the donor was
correct, or that it recovered a missing reasoning process. We use the raw margin
change rather than dividing by a potentially tiny donor/recipient baseline gap.

## Checks before interpreting an effect

| Comparison | What it checks |
|---|---|
| Unpatched donor and recipient | Whether the two inputs produce different answers or logits at all |
| Self-patch at every selected layer | Whether capturing and reinserting the recipient state preserves its output |
| Same-rule donor | Whether another visual variation causes a change without changing the rule |
| Reverse donor and recipient | Whether the effect depends on direction |
| Three X/Y/Z assignments | Whether the result follows a particular answer label |
| Final-block cross-patch | Whether replacing the final answer-position state transfers the donor's restricted logits |

Self-patches should reproduce recipient logits. At the last block, the remaining
normalization and answer head operate on the replaced token, so a cross-patch
should approximately reproduce donor logits. Final-layer transfer is therefore
an implementation sanity check, not evidence of Boolean reasoning. These tests
use a declared absolute logit tolerance of 0.05 by default. They save the actual
maximum errors so numerical differences can be inspected, rather than hidden.
A failed check stops the scan after saving diagnostics; do not relax tolerance
merely to obtain a passing experiment.

The CSV flags identical baseline predictions, nearly identical restricted logits,
and nearly identical donor/recipient rule margins. It also saves the maximum
baseline probability difference. The console reports equal baseline predictions
and near-equal logits. If the baselines are alike, a small patching effect is
uninformative about recovery. Even a large effect could transfer answer bias,
visual features or a diagnostic-scene pattern rather than an abstract rule.

## Running in Colab

Use a fresh GPU runtime, mount Drive, and clone this branch. If the checkout
already exists, fetch and switch branches instead of cloning over it.

```bash
git clone --branch feature/activation-patching https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
python -m pip install -r three-operand-experiments/code/requirements-qwen35.txt matplotlib
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data'
export SCRIPT='three-operand-experiments/code/03-activation-patching/patch_rules.py'

# CPU checks only; no model loading or output files.
python "$SCRIPT" --data-dir "$DATA" --validate-only

# One pair, one label assignment, both directions, first/middle/last blocks.
python "$SCRIPT" --data-dir "$DATA" --smoke-test

# Default full scan: one matched pair for each of the three rule pairs.
python "$SCRIPT" --data-dir "$DATA" --pairs-per-rule-pair 1
```

Pass the actual existing `--data-dir` if your images are elsewhere. The results
root defaults to the existing Drive directory:

`/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results`

If Drive is mounted differently, pass its actual existing root with
`--experiment-dir`. A missing root produces an error; the script never creates
one. New outputs go under `activation-patching/qwen3_5/`, as requested for this
experiment, with smoke outputs isolated in its `smoke/` subfolder. Nothing is
copied or synchronized to GitHub, and earlier results are untouched.

The smoke test makes four unpatched baseline passes and 18 patch passes for a
model with at least three blocks. The default full scan makes 36 baseline passes
and 54 patch passes per decoder layer. It may be slow: each patch is a fresh
ten-image forward pass. Start with the smoke test and inspect its numerical
checks before committing Colab time to the full scan. `--pairs-per-rule-pair`
accepts 1–6, but extra pairs still reuse this small dataset.

Rerun the same full command after a disconnection. Completed baseline states and
patch rows are reused. `config.json` checks the dataset hashes, model settings,
prompts, pairs, label assignments, software versions and tolerance before resume.
Changing these settings requires a separate existing results root, rather than
mixing checkpoints. If a numerical control failed, resume refuses to skip it:
inspect the saved evidence, correct the setup, and preserve the failed run before
starting a fresh one.

## Outputs and current status

`baselines.csv` contains unpatched X/Y/Z and rule-mapped logits and probabilities
for both rules and their same-rule donor variants. `patches.csv` contains each
layer, direction, label layout and control, plus the baseline and patched values,
margin changes and numerical-check flags. The `donor_rule` column names the
other rule in the cross-rule comparison for all control types;
`patch_source_rule` and `patch_source_scene` identify the state actually inserted.
This keeps the same margin definition when plotting self and same-rule controls. `patching_effects.png` plots the mean
margin change by block for cross-rule, same-rule and self-patches. The averaging
is descriptive across reused inputs and label assignments; it is not an estimate
from independent trials.

The `.pt` files store unpatched final-token states and logits, and `patches.jsonl`
is an atomic checkpoint written after each completed patch. `numerical_checks.json`
records self/final check counts, failures and tolerance. Temporary files are
renamed only after writing completes. All of these outputs stay on Drive.

CPU tests validate pair construction, output cloning, hook cleanup, dtype
conversion, different sequence lengths, self/final transfer on a toy model and
checkpoint resume. Those checks do **not** establish that Qwen3.5 GPU patching
works. Findings remain pending until the Colab experiment runs and its controls
are inspected. A larger, independently generated image pool and more varied
diagnostic constructions would be needed before drawing general conclusions.
