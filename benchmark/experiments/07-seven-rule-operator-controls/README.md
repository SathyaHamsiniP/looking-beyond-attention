# 07 — Can Qwen find the rule behind the pictures?

In our earlier experiments, Qwen had to choose between three possible rules. But a correct answer did not necessarily mean it had worked out the pattern in the pictures. It might simply have preferred certain answers.

So we tried the task with seven rules.

Each question shows ten pictures. A green cube (A), blue sphere (B), and red cylinder (C) may or may not appear in each one. All ten pictures follow the same hidden rule, and Qwen has to pick that rule from seven options.

For example, `(A AND B) OR C` means that a picture qualifies if it has the red cylinder, or if it has both the green cube and blue sphere. Other rules combine the objects differently, including rules using XOR (one of two objects, but not both). The complete list is in [`rules_7.json`](../../code/00-common/rules_7.json).

## What we tried

We gave the pretrained Qwen3.5-4B model 49 sets of ten pictures. For each set, we changed the order of the seven answer options and asked again. This matters because a model that understands the rule should not depend on where the correct option appears.

We used the saved validation examples and did not train the model further.

## What happened

Qwen got **48 out of 343 answers right (about 14%)**. With seven choices, random guessing would also give about 14%.

The more interesting result was *which options it picked*. It chose option 7 on 261 questions and option 6 on the other 82. It **never chose options 1–5**. Its answer also changed when we reordered the choices for the same pictures.

![Which options Qwen chose, and which rules it confused](../../results/seven-rules/qwen3.5-4b/operator-controls/answer_bias_and_confusion.png)

This gives us little evidence that Qwen was identifying the rule from the pictures. Its preference for the last options seems to have played a much bigger part in its answers.

The earlier three-rule experiments also showed answer-position bias. But their image sets were different, so we cannot use these scores alone to say exactly how much harder seven rules are than three.

The [full results](../../results/seven-rules/qwen3.5-4b/operator-controls/summary.json) and [individual predictions](../../results/seven-rules/qwen3.5-4b/operator-controls/predictions.csv) are available in the results folder.

## Running the analysis

The results are already saved. To regenerate the summary and graph, you need the original image dataset, saved dataset manifest, and pretrained validation predictions. These large inputs are not included in Git.

From the repository root:

```bash
pip install -r benchmark/code/00-common/requirements-seven-rule-controls.txt
python benchmark/code/07-operator-controls/operator_controls_7rules.py --manifest PATH_TO_MANIFEST --data-dir PATH_TO_DATA --baseline PATH_TO_PRETRAINED_VALIDATION --out PATH_TO_OUTPUT
```

Replace the paths with your local copies. Keep the output folder outside the repository. This command checks saved predictions and creates the summary and graph; it does not run the model again. The published predictions were produced before the code reorganization, so their source hashes require the [original matching checkout](../../code/README.md#earlier-saved-runs). Use the moved runner for new, matching inputs.

**What this tells us:** Getting a rule-selection question right is not enough to show logical reasoning. We also need to know whether the model is actually using the examples, rather than relying on the way the question is presented.
