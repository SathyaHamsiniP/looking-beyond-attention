# Verified baseline findings

Checked 3,456 saved trials. Inputs match the saved fingerprint.

At layer 32, all 54 final-token patches reproduce the donor prediction. Of 36 different-rule trials, 16 change the answer and seven change it to the donor true rule.

| Condition at layer 32, final token | Trials | Answer changes | Mean signed change (percentage points) | Mean absolute change (percentage points) |
|---|---:|---:|---:|---:|
| same | 18 | 6 | 0.3679 | 7.0356 |
| different | 36 | 16 | -1.7147 | 7.1304 |

Layer 6 has the highest signed different-rule final-token mean: -0.04659 percentage points. It is negative. No layer passes the original 0.05 control cutoff.

A donor-prediction match is not necessarily correct. Same-rule patches also change answers. Similar absolute changes across conditions do not identify a rule-specific reasoning layer.

Plots count reused pairs and rotations, not independent image sets. The confusion matrix counts each evaluated set once per rotation, including donor-only sets. Answer bias therefore describes 27 original set/rotation predictions, not 3,456 independent original answers.

Only two distinct receiver/patched rule probabilities can be reconstructed in different-rule records. Same-rule missing probabilities stay unknown. See examples.json for explicit reconstructed-rule labels. The two examples are the first matching rows in file order; they are not a representative statistical sample.

Source files are copied byte-for-byte into raw/. No model was rerun. Read the parent guide for the beginner explanation.

## How to interpret the changes across all 32 layers

The **signed mean** averages `P_after(donor true rule) - P_before(donor true rule)`. Positive and negative trials cancel. The **mean absolute change** averages the *sizes* of those shifts, regardless of sign. The **answer-change rate** counts changes to the winning option; it does not measure correct rule transfer.

For example, +2 and −4 percentage-point shifts give a **−1 pp signed mean** but a **3 pp mean absolute shift**.

| Layer | Different-rule final-token signed mean | Mean absolute shift | Answer changed | Changed to donor true rule |
|---|---:|---:|---:|---:|
| 1 | −0.0484 pp | 0.34 pp | 0/36 | 0/36 |
| **6** | **−0.0466 pp** | **1.20 pp** | **1/36** | **0/36** |
| 19 | −0.2912 pp | 3.97 pp | 8/36 | 1/36 |
| 23 | −0.8286 pp | 4.83 pp | 7/36 | 1/36 |
| 31 | −1.4884 pp | 5.11 pp | 10/36 | 4/36 |
| **32** | **−1.7147 pp** | **7.13 pp** | **16/36** | **7/36** |

Layer 6 is **not** the largest shift. Its signed mean is merely the **least negative** of the 32 different-rule final-token layer means. There were **15 positive, 20 negative and one effectively unchanged** donor-rule probability shifts among its 36 patches. Their near-cancellation makes the mean −0.0466 pp, even though the mean absolute shift is 1.20 pp. **Every one of the 32 layer-wise signed averages is negative** for this condition. Some individual trials were positive, but there is no layer with a *positive average* toward the donor's true rule.

Early layers generally have smaller effects; later layers tend to change probabilities and answers more, though not strictly at every layer. The largest absolute shift and answer-change rate occur at layer 32, where replacing the final-token vector is directly before the answer readout. This may transfer an **incorrect donor prediction**; it does not localize Boolean reasoning. The average absolute shift at layer 32 is **7.13 pp for different-rule patches and 7.04 pp for same-rule patches**.

The data establishes **what changed**, not the internal reason for the negative averages. Answer-label bias, donor mistakes, varying images, and opposing trial effects are possible contributors; the aggregate does not isolate one. Before choosing a layer for single-layer LoRA, explicitly define “shift,” compare against same-rule and previous-token controls, and select candidates on validation data. No layer has been identified as a rule-computation site.

For the fuller intuition and context, read [the Experiment 03 guide](../../README.md#reading-the-layer-wise-results-what-actually-shifts).

