# Experiment 04 extension: three-operand operator controls

## Research question

Can a vision-language model infer which three-operand Boolean expression generated ten positive CLEVR-style scenes? Can it use decisive evidence across scenes rather than rely on answer-label or option-position cues?

## Concepts and candidate rules

- `A`: green cube
- `B`: blue sphere
- `C`: red cylinder

The rules are `R1 = (A AND B) OR C`, `R2 = (A AND C) OR B`, `R3 = (B AND C) OR A`. All concept matches require the **exact shape and color**; other visible objects are distractors.

The `coverage` sampler includes all **five positive three-bit assignments** satisfying each rule, including a *one-hot* scene that reveals the variable outside the AND. This guarantees unique identification of the generating rule among the three candidates. In contrast, `iid_positive` sampling is with replacement and may leave the rule ambiguous.

## Main evaluation

**3 generating rules × 8 sets of 10 images per rule × 3 cyclic X/Y/Z label assignments = 72 scored presentations per model**, built from 24 distinct ten-image sets. Seed: **2026**. Input pool: variations 0–3 of the 48-scene dataset (`data/metadata.json` and `data/images/`).

The score is the model's **next-token logit over X, Y, Z, normalized over those three letters**. It is not a full-response likelihood, and strong label/position preferences can produce apparently correct answers without inference of the rule.

## Controls and interventions

| Experiment flag | Purpose | Scope |
|---|---|---|
| `main_72` | Infer the hidden rule from ten images | Both models |
| `text_only` | Replace images with exact A/B/C scene descriptions | Both |
| `captioned` | Provide images together with ground-truth scene descriptions | Both |
| `position_order` | Change display order while keeping labels associated with rules | Both |
| `label_assignment` | Change X/Y/Z label-to-rule assignments | Both |
| `iid_positive` | Test positive-only examples without guaranteed complete coverage | Both |
| `ablation` | Remove a decisive one-hot image vs a non-decisive control image | Both |
| `perception` | Check concept recognition on eight held-out images | Both |
| `gates` | Evaluate explicit intermediate AND and final OR gates | Both |
| `discovery_open` | Generate the rule as an expression without options | Qwen3-VL only |
| `discovery_guided` | Identify decisive scene, outside variable and expression | Qwen3-VL only |

### Decisive-image ablation

For each of 24 `decisive_once` sets, measure the true-label probability with **all 10 images**, **without the unique one-hot scene**, and **without a non-decisive scene** (72 model calls total). Removing decisive evidence deliberately makes the remaining set ambiguous. The meaningful outcome is the *change in probability*, **not** whether the model still happens to choose the generating rule.

In the saved Qwen3-VL run, the mean true-label probability difference (full minus decisive-removed) was approximately **+0.0000036**, compared with **−0.0008926** for the control removal (mean paired difference **+0.0008962**). This is small and must be interpreted in the presence of substantial option bias.

## Saved findings (October 2026)

| Task | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Main task (72) | 33.3% | 31.9% |
| Text only (24) | 33.3% | 29.2% |
| Captioned (24) | 33.3% | 12.5% |
| Position order (144) | 32.6% | 34.7% |
| Label assignment (144) | 32.6% | 27.8% |
| IID positive (24) | 33.3% | 25.0% |
| Perception (8) | 100% | 100% |
| Both gates correct (24) | 79.2% | 95.8% |

The three-way chance baseline is **33.3%**. Qwen3-VL chose the **first displayed option in 143/144** position-order runs. The results do not establish that either model inferred the generating rule above chance, nor that attention reflects its causal computation.

**Qwen3.5 free-text discovery caveat:** Open/guided generations in the original experimental session were truncated before reliably parseable answers. They are excluded from the Qwen3.5 maintained script/archive rather than scored as reasoning failures. The Qwen3-VL script retains its completed discovery controls.

## Executable implementation and provenance

- **Qwen3-VL:** `../../code/exp04_qwen3vl.py` with Roshan's separately supplied original wrapper via `--upstream`
- **Qwen3.5:** `../../code/exp04_qwen35.py` with native Transformers 5.x adapter
- **Requirements:** `../../code/requirements-qwen3vl.txt`, `../../code/requirements-qwen35.txt`
- **Runner instructions:** [main README — setup and execution](../../../README.md#reproduce-on-your-machine-or-a-gpu-server)
- **Original records:** `../../results/<model>/<condition>/config.json`, `runs.jsonl`, and `summary.json`

The scripts are the maintained implementation; earlier Colab notebooks are no longer part of the published repository. The Qwen3.5 Python runner completed a real Colab GPU smoke test and a new 72-presentation run reporting **31.9%**. The converted Qwen3-VL runner passed CPU tests but has not yet had a fresh GPU smoke test. The original Qwen3-VL inference records remain archived.

Comparisons should account for the adapters differing in model-specific preprocessing and chat formatting. Repeated label permutations are grouped by ten-image set for the permutation/bootstrapped analyses; they are not independent samples.
