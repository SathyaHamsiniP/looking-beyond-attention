# Looking beyond Qwen's answers

When we ask a vision-language model to find a logical rule from pictures, what does a correct answer really tell us? And if it gives a wrong answer, has it understood nothing at all?

These benchmark experiments explore those questions with **Qwen3.5-4B**. In each question, pictures contain different combinations of a green cube (A), blue sphere (B), and red cylinder (C). All the pictures follow one hidden Boolean rule, and Qwen has to identify it. We first worked with three possible rules, then tried seven.

## Experiments

| Experiment | Question | What we found with seven rules |
|---|---|---|
| [07 — Operator controls](experiments/07-seven-rule-operator-controls/README.md) | Does Qwen actually identify the rule, or favor certain answer positions? | **48/343 correct (14.0%)**, about chance. It picked only the last two options. |
| [08 — Activation patching](experiments/08-seven-rule-activation-patching/README.md) | Can changing what happens inside Qwen change the rule it chooses? | The swaps changed answer probabilities, but not in a way that proves the rules were being used. |
| [09 — Probing](experiments/09-seven-rule-probing/README.md) | Is rule-related information present inside Qwen even when its answer is wrong? | Final-layer probes scored **30.6%, 38.8%, and 20.4%** with one, five, and ten images; Qwen itself scored about 14.3%. |
| [06 — LoRA](experiments/06-seven-rule-lora/README.md) | Would additional training help? | No verified full-run seven-rule evaluation is available yet. |

The experiments are connected, but answer accuracy, probe accuracy, and activation patching test different things. A probe doing well does not mean Qwen reasons correctly; a change after patching does not prove rule information was transferred.

### Reading the results

The strongest lesson so far is that **the model's answer does not tell the whole story**. It has a strong habit of choosing certain answer positions. At the same time, a separately trained probe can sometimes identify the correct rule from information inside the model. Whether that information supports actual logical reasoning remains open.

The five-image probing condition is especially important to interpret carefully: those images were chosen knowing the correct rule. Also, images repeat across some sets, and most of the seven-rule experiments use validation data rather than an untouched final test. Comparing three-rule and seven-rule scores directly would be misleading because the setups differ.

Find the saved results, CSVs, graphs, and run records in the [results folder](results/README.md). The code for each experiment is in this `benchmark/` directory. To run local checks:

```bash
PYTHONPATH=benchmark python -m unittest discover -s benchmark -p 'test_*.py'
```

Large image sets and model checkpoints are not included in the repository. The seven-rule probing archive also contains frozen feature arrays: keep their original metadata with them if importing or reusing the saved run.
