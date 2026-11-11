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
