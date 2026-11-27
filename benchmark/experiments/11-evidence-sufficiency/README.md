# 11 — When is there enough evidence?

**Implemented. Awaiting GPU evaluation.**

Earlier tasks asked for one rule. But a picture with all three objects fits six rules. Picking one of them discards five answers that the evidence still allows. Here the target is the complete set of rules that fit every picture.

## Follow one chain

A, B and C mean a green cube, a blue sphere and a red cylinder are present. Each row below adds one scene. A 1 means the object is present. The solver keeps a rule only if every scene so far is in its truth table.

| Scene added | Rules still possible |
|---|---|
| 111 | R1, R2, R3, R4, R5, R6 |
| 011 | R1, R2, R3, R4, R5 |
| 001 | R1, R4, R5 |
| 110 | R1, R5 |
| 101 | R1 |

The runner searches the seven saved truth tables for sets with a unique answer. It selects an endpoint for each rule and orders its evidence to cover different levels of ambiguity. Every prefix is a separate question. It can keep several rules, eliminate some, or leave the answer unchanged. The default design has seven chains and 36 prefixes. It covers every candidate count from one to six. The empty set is excluded. No rule accepts 000.

Qwen sees the pictures, the meanings of A/B/C and the seven expressions. It selects all options consistent with every scene. The symbolic control supplies the same evidence as explicit A/B/C values. Each question starts a fresh conversation. All seven cyclic orders move each rule through every position. Only the evaluator sees filenames, bit annotations for the visual task and the correct answers.

## Read the measurements

Exact match requires all and only the valid rules. Precision measures how many selected rules fit. Recall measures how many valid rules survived. The report includes both by rule and averaged across presentations. It groups exact match by candidate count. A malformed list scores zero on exact match and on the presentation-level precision and recall. Its raw text is saved.

The elimination score asks whether all newly excluded rules were present in the prior answer and absent in the next one. The reintroduction rate asks whether a later valid response brings back a rule previously absent from both the evidence-based answer and a parsed prediction. Each rate has its own denominator. Order agreement compares decoded rule sets for the same input. Repeated orders and prefixes share evidence. The paired model and modality summaries weight each chain equally.

A correct symbolic answer paired with a wrong visual answer narrows the difficulty to the visual version of the task. Both versions also require following a list-format instruction. The saved adapter was trained to give one answer, so a tendency to select just one option matters here. These tests measure behavior on this rule family.

## Run in Colab

Use the [shared Colab commands](../../code/README.md#run-experiments-11-and-12-in-colab) with experiment 11. The [saved design counts](../../results/seven-rules/qwen3.5-4b/evidence-sufficiency/design.json) describe verified inputs. Model measurements will be written to your output directory.
