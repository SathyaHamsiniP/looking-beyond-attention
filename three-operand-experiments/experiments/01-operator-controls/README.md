# 01 — Can the models identify a rule from examples?

We wanted to understand whether recognizing objects in images was enough for a
model to work out a rule connecting them. We extended the earlier two-operand
task to three operands so that the model had to consider both the grouping of
concepts and two Boolean operations.

A is a green cube, B is a blue sphere and C is a red cylinder. A concept is
present only when both colour and shape match; other objects are distractors.
We use these three candidate rules:

| Rule | Expression |
|---|---|
| R1 | (A AND B) OR C |
| R2 | (A AND C) OR B |
| R3 | (B AND C) OR A |

Our dataset contains 48 images: the eight A/B/C configurations, with six visual
variations each. The existing `data/metadata.json` records their configurations.
The main experiment samples ten positive images with coverage of all five
positive configurations for a rule. We use eight sampled sets for each of three
rules and three cyclic answer-label assignments. That gives 72 trials from 24
underlying scene sets, not 72 independent sets. The sets also reuse images.
Variations 0–3 supply the sampling pool; variations 4–5 are held out.

The model chooses X, Y or Z. We normalize its next-token scores over those three
options and map the answer back to R1, R2 or R3. This measures relative choice
probabilities, not probabilities over every possible output.

## What we tried and what happened

We replaced images with exact text descriptions, supplied both images and text,
changed option order, and reassigned answer labels. We also sampled positives
without guaranteeing that they uniquely identified a rule. Separate controls
asked whether the model recognized the three concepts and could evaluate the
Boolean operations when given explicitly. Qwen3-VL additionally has open-ended
and guided discovery runs; the maintained Qwen3.5 runner excludes these because
truncated generations prevented reliable final-answer evaluation.

The recorded summaries in [results](../../results/01-operator-controls/) give:

| Task | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Main rule identification | 33.3% | 31.9% |
| Text-only | 33.3% | 29.2% |
| Images and descriptions | 33.3% | 12.5% |
| Option-order control | 32.6% | 34.7% |
| Answer-label control | 32.6% | 27.8% |
| Independently sampled positives | 33.3% | 25.0% |
| Object recognition (8 images) | 100% | 100% |
| Both Boolean gates correct (24 cases) | 79.2% | 95.8% |

Rule-identification chance is 33.3%. Both models did well on the small recognition
test and much better on explicit gate evaluation, but neither reliably inferred
the rule from examples. Answer presentation mattered: Qwen3-VL chose the first
position in 143 of 144 option-order trials.

We also removed the image that distinguished the rule, then compared this with
removing a non-decisive image. The recorded mean drops in the correct-rule
probability were:

| Removal (24 sets) | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Decisive image | 0.00000363 | -0.01044 |
| Non-decisive image | -0.00089260 | 0.00562 |

These are probability differences, not percentages; negative means the
probability increased after removal. They show little average change, especially
for Qwen3-VL, rather than a clear sensitivity to decisive evidence. Answer biases
and the small shared image pool limit what we can infer from this.

We were left wondering whether useful information existed internally even when
the final answer was wrong. That led to [experiment 02](../02-probing/README.md).
Good recognition and explicit gate evaluation do not by themselves establish
successful rule induction.

## Running these controls

Use Colab with a GPU and mounted Drive. From `/content/looking-beyond-attention`,
install the selected model's shared requirements. Use a fresh runtime for each
model because the Transformers versions differ. Colab supplies CUDA PyTorch.
The commands below assume the existing images and metadata are in the project's
Drive `data/` directory; use their actual existing path if different.

```bash
python -m pip install -r three-operand-experiments/code/requirements-qwen35.txt
python three-operand-experiments/code/01-operator-controls/operator_controls_qwen35.py \
  --data-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data' \
  --experiment-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results' \
  --precision 4bit --smoke-test
```

For Qwen3-VL, extract the supplied reference repository separately first:

```bash
python -m pip install -r three-operand-experiments/code/requirements-qwen3vl.txt
python three-operand-experiments/code/01-operator-controls/operator_controls_qwen3.py \
  --upstream '/content/roshan/looking-beyond-attention-main' \
  --data-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data' \
  --experiment-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results' \
  --precision 4bit --smoke-test
```

Replace `--smoke-test` with `--experiment main_72` for the main experiment; use
`--help` for the other controls. `--validate-only` checks data and sampling
without loading weights. All new outputs stay on Drive under `operator-controls/qwen3/` or
`operator-controls/qwen3_5/`. The results root defaults to the path above and
must already exist; `--experiment-dir` overrides it when Drive is mounted
differently. No results are copied to GitHub. Archived results remain unchanged.

## Implementation credit

The Qwen3-VL runner reuses Roshan and collaborators' external wrapper and
next-token readout. Qwen3.5 uses Hugging Face through our maintained adapter.
Their preprocessing paths differ, so model comparisons include that limitation.
The folder cleanup preserves the previous prompts, model settings, calculations,
images and result-file contents. The old internal version strings in archived
configurations are retained as provenance, not used as our experiment numbering.
