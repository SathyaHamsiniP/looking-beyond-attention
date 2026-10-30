# Three-Operand Boolean Rule Identification

## Motivation

Suppose a model is shown several images that all satisfy the same logical rule. Can it work out what that rule is?

More importantly, if it gets the answer right, is it because it used the relevant evidence, or because it preferred a particular answer regardless of the images?

We designed these experiments to investigate that distinction using two vision-language models: **Qwen3-VL-4B-Instruct** and **Qwen3.5-4B**.

## Experimental setup

We use three visual concepts:

- **A:** Green cube
- **B:** Blue sphere
- **C:** Red cylinder

The model must distinguish between three candidate expressions:

| Rule | Boolean expression |
|---|---|
| R1 | (A AND B) OR C |
| R2 | (A AND C) OR B |
| R3 | (B AND C) OR A |

The images also contain distractor objects. A concept counts as present only when both its shape and color match.

For each rule, we create sets of ten images that satisfy the expression. Each set includes enough distinct examples to identify its generating rule among the three candidates.

We generated 48 base images covering the eight possible combinations of A, B, and C, with six visual variations per combination.

## Main experiment

The model receives ten images and is asked to select the generating rule from three options labeled X, Y, and Z.

We evaluate:

- 3 generating rules
- 8 independently sampled image sets per rule
- 3 different assignments of answer labels

This gives **72 trials per model**, based on 24 distinct image sets.

We compare the model's next-token scores for X, Y, and Z to determine its prediction. The reported probabilities are normalized over these three choices.

## What else did we test?

Simply measuring accuracy wouldn't tell us why the model succeeds or fails. We therefore added several controls.

| Experiment | What we wanted to understand |
|---|---|
| `text_only` | Does replacing images with exact descriptions make rule identification easier? |
| `captioned` | Does providing both images and descriptions help? |
| `position_order` | Does the model favor an option because of where it appears? |
| `label_assignment` | Does changing the X/Y/Z labels change its decisions? |
| `iid_positive` | What happens when examples are sampled without guaranteeing that they uniquely identify the rule? |
| `ablation` | Does removing a decisive image affect the prediction more than removing another image? |
| `perception` | Can the model recognize A, B, and C correctly? |
| `gates` | Can it evaluate the Boolean expressions when the operations are given explicitly? |

For Qwen3-VL, we also tested open-ended rule identification and guided reasoning without multiple-choice options.

### Removing decisive evidence

One experiment we found particularly interesting was removing the image that distinguishes the correct rule from the alternatives.

We compared the model's prediction using all ten images against its prediction after removing that decisive image. We also removed a non-decisive image as a control.

If the model is relying on the distinguishing evidence, we would expect removing it to have a meaningful effect on its prediction.

For Qwen3-VL, the average change in the probability assigned to the correct rule was extremely small. This suggests limited sensitivity to the decisive image in our setup, although the strong answer bias makes it difficult to isolate the cause.

## Results

| Experiment | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Main rule identification | 33.3% | 31.9% |
| Text-only | 33.3% | 29.2% |
| Images with descriptions | 33.3% | 12.5% |
| Option-order control | 32.6% | 34.7% |
| Label-assignment control | 32.6% | 27.8% |
| Independently sampled positive scenes | 33.3% | 25.0% |
| Object recognition (8 images) | 100% | 100% |
| Both Boolean gates correct (24 cases) | 79.2% | 95.8% |

The chance accuracy for rule identification is 33.3%.

### What these results suggest

**1. Object recognition was not the main obstacle in our tests.**

Both models recognized the target concepts correctly in the eight-image perception check.

**2. Evaluating a rule is easier than discovering one.**

When the expressions were explicitly provided, the models were much better at computing their outputs than at identifying the expression responsible for a set of examples.

**3. Answer presentation matters considerably.**

Qwen3-VL overwhelmingly preferred the first displayed option in our position-order control. Qwen3.5 also showed substantial answer preferences, although the pattern differed.

**4. More capable Boolean execution did not guarantee better rule identification.**

Qwen3.5 performed better on our explicit Boolean gate test, but its hidden-rule identification accuracy remained close to chance.

Together, these results motivate a closer examination of how models combine information across examples, rather than treating correct object recognition or individual logical computations as evidence of successful rule inference.

## Code and data

The experiments are implemented in:

- `code/exp04_qwen3vl.py`
- `code/exp04_qwen35.py`

The corresponding requirements files are in `code/`, and the 48 images and their metadata are in `data/`.

Recorded experiment outputs are available in `results/`, including individual predictions, configurations, and summaries.

The Qwen3-VL runner uses an external model wrapper from the original operator-controls implementation. The Qwen3.5 runner uses the Hugging Face implementation directly. Because the models use different preprocessing and inference code, comparisons should be interpreted with that distinction in mind.

## Notes

The 72 main trials reuse 24 underlying image sets with different answer-label assignments, so they should not be treated as 72 independent samples.

The Qwen3.5 open-ended discovery experiments are not included in the maintained implementation because truncated generations prevented reliable evaluation of their final answers.

This work extends the operator-control experiments in Roshan and collaborators' original research code.

The results reported here reflect the specific models, prompts, scenes, and evaluation conditions used in this study.
