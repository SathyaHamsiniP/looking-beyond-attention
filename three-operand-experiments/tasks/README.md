# Observation tasks

V1 preserves the positive-only three-rule question. The historical runner remains unchanged; new V1 manifests cover all five satisfying configurations. In V1, one positive singleton configuration reveals the rule: 001 for R1, 010 for R2, and 100 for R3. Accuracy is therefore not evidence that the whole expression was evaluated.

V2 uses the same three rules and ten labelled observations. Every set has five true and five false labels. Each observation leaves at least two possible rules, while all observations together leave exactly the generating rule. For these three rules, the identifying evidence must combine negative diagnostic observations. The five shared truth-table rows alone cannot distinguish the candidates.

V3 uses eight distinct rules. It includes different AND/OR parentheses, XOR, and negation. This is a selected, finite candidate library, not a benchmark of arbitrary logical expressions. All eight rules can form balanced, singleton-safe, uniquely solvable sets. Generation rejection-samples sets until these invariants hold, with a bounded failure and a clear error. We hold the positive count fixed across rules. Configuration frequencies can still provide shortcuts; inspect each set's saved counts. No claim of unseen logical structure generalization is supported.

## Truth tables

Bits are ordered A/B/C. Expressions use explicit Boolean functions; no expression from a manifest is executed.

| Rule | Expression | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---|---|---|---|---|---|---|---|---|
| R1 | (A AND B) OR C | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| R2 | (A AND C) OR B | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| R3 | (B AND C) OR A | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| R4 | (A OR B) AND C | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| R5 | (A XOR B) OR C | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| R6 | (A AND NOT B) OR C | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 |
| R7 | A XOR (B OR C) | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| R8 | (A OR C) AND NOT B | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |

## Splits and manifests

The original 800 training images stay in training. The 25 held-out images per Boolean configuration are deterministically shuffled with seed 2026 and divided 12/13 or 13/12 between validation and final-test. The result is exactly 800/100/100. The derived mapping is checked in at `splits/dataset-v1-seed2026.json`; source metadata and images are untouched. Different image variations are held out; the Boolean configurations and rule library are shared across partitions.

Each manifest embeds that split, metadata hash, published ZIP checksum, exact rules/tables, seed, option order, per-observation image path/bits/label, set ID, and validation report. Within-set duplicates are rejected. Across-set image reuse within a partition is recorded; such sets are not independent observations. No image crosses partitions.

Run `build_sets.py --help` and `validate_sets.py --help` or follow [the Colab runbook](../COLAB.md). Config JSON files document the standard recipe; CLI arguments select task, count, size, and seed. The default is ten images, twelve sets per rule per partition. Larger tasks require even sizes and explicit recording. Generated manifests belong on Drive, not alongside tracked source.

The validator recomputes the symbolic oracle answer and all singleton checks against original metadata. Oracle accuracy of 1.0 means the generated problem has a unique symbolic answer; it is not a model result. Token mappings are checked at runtime against the actual tokenizer; complete label-sequence likelihoods are used if labels span multiple tokens. All cyclic option orders are evaluated by default. This controls position but does not exhaust every permutation.
