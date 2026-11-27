# 12 — Does the surrounding evidence change the answer?

**Implemented. Awaiting GPU evaluation.**

One picture could cue the right answer in a rule-identification task. We ask the model to answer two matched questions separately. It never sees the other member or the pair type.

| Pair | What changes? | What should happen? |
|---|---|---|
| Replacement | One image and its A/B/C configuration | Switch to the other unique rule. |
| Context | At least two configurations around the very same focal image | Switch even though the focal pixels and position stay fixed. |
| Control | One image is replaced with a different rendering of the same configuration | Keep the same rule. |

The solver checks every subset of the eight configurations. Only sets with a unique answer enter the pair search. Members have equal length. Shared configurations keep their image file and position. For example, replacing 001 with 010 in **111, 011, 101, 110, 001** changes the unique answer from R1 to R2.

Here is a stronger pair where the fourth picture is identical in both questions.

| Full context | Unique answer |
|---|---|
| 111, 011, 100, **101**, 010 | R4 |
| 111, 011, 110, **101**, 001 | R1 |

All 21 unordered rule pairs support replacement. Eighteen support the stronger context condition. R1, R2 and R3 each have only one unique configuration set. Those sets differ from one another in just one place. Each feasible rule pair appears once per repeat in its condition. Each rule gets one same-answer control. This balances feasible contrasts. R1–R3 appear less often in the context condition. The [original 32-image design audit](../../results/seven-rules/qwen3.5-4b/context-counterfactuals/design.json) lists the counts.

## What Qwen receives

The visual task shows positive scenes, the object definitions and seven numbered expressions. It asks for one number. The symbolic task replaces each image with its A/B/C values. It uses the same rule order. Control members have identical symbolic inputs. The position of the focal picture is unmarked. Seven cyclic orders cover every answer position.

The default pool contains 32 variants for each of the eight configurations. The 46 pairs need 300 image slots after counting each shared file once per pair. The allocator uses 224 distinct files and fills 76 slots by reuse across pairs. Those shared files occur in 29 of the 46 pairs. This design needs up to 47 variants per configuration (376 balanced images) to make all pairs image-disjoint. The 000 configuration is never used because no rule accepts it.

## Read the measurements

Check each member's accuracy, both-correct accuracy within each pair, and switches between parsed answers. A **correct switch** requires both target rules in the right direction. A switch on a same-answer control is an error. Malformed answers fail accuracy. Both answers must parse for the raw switch rates. Order agreement compares rule identities after decoding the option numbers. Model and modality differences are paired within the same pair, averaging over its members and orders.

Two shortcuts receive perfect object annotations. One uses only the focal configuration (the first image when no focal image is designated). It picks the first compatible rule. It must give the same answer for both context members. The other compares the three object frequencies with those expected under uniform sampling from each rule's positive configurations. It picks the closest by squared distance, breaking ties in rule order.

Correct answers on both context members rule out reliance on that focal image alone. Another picture, a count or a combination of simple cues may still identify the answer. High switch rates alone are weak evidence. Read them alongside both-correct accuracy, control errors and the shortcut scores.

The report counts valid correct, valid incorrect and invalid-format answers separately. The 95% intervals resample whole pairs and keep all their presentations together. Shared configurations and any shared images limit the interpretation.

## Run in Colab

Follow the [shared Colab commands](../../code/README.md#run-experiments-11-and-12-in-colab) with experiment 12. The default design has 46 pairs and 92 members. Use a small prediction limit first.
