# Experiment 04 extension: three-operand operator controls

## Question
Can a vision-language model infer which three-operand Boolean expression generated ten positive CLEVR-style scenes?

## Logic
- `R1`: `(A AND B) OR C`
- `R2`: `(A AND C) OR B`
- `R3`: `(B AND C) OR A`

A is green cube, B is blue sphere, C is red cylinder. A sampled **coverage** set has all five positive assignments for its rule, including the decisive one-hot scene, and therefore uniquely identifies the generating rule among these three options. Images have distractors.

## Main experiment
Three rules × eight independently sampled 10-image sets per rule × three cyclic label assignments = **72 scored presentations**. Model outputs are restricted next-token candidate probabilities over X/Y/Z. This measures the response distribution under controlled prompts, not unconstrained reasoning ability.

## Controls
- `text_only`: use ground-truth Boolean scene descriptions.
- `captioned`: provide descriptions and images.
- `position_order`: permute option presentation order.
- `label_assignment`: change letters assigned to expressions without reordering expressions.
- `iid_positive`: sample positive scenes with replacement; samples may lack a uniquely informative assignment.
- `ablation`: remove decisive versus nondiscriminating examples and compare true-label probability.
- `perception`: check eight held-out single images.
- `gates`: execute explicitly given Boolean rules on all eight assignments.
- `discovery_open` / `discovery_guided` (Qwen3-VL only): free-text rule derivation. The Qwen3.5 counterparts were inconclusive because responses were truncated before parseable final answers. They have been excluded from this cleaned repository; the original results ZIP remains archived in the parent Google Drive folder.

## Results observed (October 2026)
| Task | Qwen3-VL-4B | Qwen3.5-4B |
| --- | ---: | ---: |
| Main 72 | 33.3% | 31.9% |
| Text only | 33.3% | 29.2% |
| Captioned | 33.3% | 12.5% |
| Position order | 32.6% | 34.7% |
| Label assignment | 32.6% | 27.8% |
| IID positive | 33.3% | 25.0% |
| Perception (n=8) | 100% | 100% |
| Both gates correct (n=24) | 79.2% | 95.8% |

The 3-way chance baseline is 33.3%. These experiments do **not** demonstrate improvement in three-operand hidden-rule inference after switching to Qwen3.5. Strong presentation-position biases complicate causal interpretations. Explicit Boolean execution improved on the small gate-check sample.

## Provenance and reproducibility
The two Colab notebooks in `notebooks/exp04/` are the authoritative executable implementations; there is no separate `code/` tree. Qwen3.5's open/guided discovery functions and corresponding result folders were excluded after truncated outputs prevented valid evaluation. Saved `config.json`, `runs.jsonl` and `summary.json` files support reproduction and analysis of the retained experiments.

Seed: 2026. Scene metadata: `data/metadata.json`. Image set: 48 (eight 3-bit assignments × six variations). Sampling pool uses variations < 4; perception checks use variation 4. A requires exact color and shape.
