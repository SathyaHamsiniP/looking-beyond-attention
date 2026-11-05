# 05 — Does activation patching change after LoRA?

## 1. Why are we doing this after Experiment 04?

In [Experiment 04](../04-lora/README.md), we give Qwen3.5-4B training examples to help it recognize which Boolean rule explains ten pictures. We check whether its answers improve on held-out picture sets.

But **better answers alone do not tell us what happened inside Qwen**.

In [Experiment 03](../03-activation-patching/README.md), we changed one internal activation vector at a time and observed whether the model's answer changed. We now repeat that idea **twice**:

1. On the original pretrained Qwen3.5-4B.
2. On the same base model with our newly trained LoRA adapter attached.

Then we ask: **does patching have a different effect after the model has been fine-tuned?** Does it become more likely to move toward the donor's correct Boolean rule, or does it mainly move toward the donor's preferred answer label?

This is **not another training stage**. Activation patching does not update model weights. The training happened in Experiment 04. Here we measure what the two model versions do.

The plan is:

~~~text
                       The SAME evaluation image sets
                                     |
                       The SAME receiver-donor pairs
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
      Original Qwen3.5-4B                         Qwen3.5-4B + LoRA
              |                                             |
              v                                             v
        Run activation                                Run activation
           patching                                     patching
              |                                             |
              v                                             v
       Save trial results                           Save trial results
              |                                             |
              +----------------------+----------------------+
                                     |
                                     v
                    Compare the layer-by-layer effects
~~~

**Status:** This new comparison has **not yet been run on pretrained Qwen**. There are no post-LoRA patching findings yet. The earlier Experiment 03 results remain a separate, completed reference.

## 2. What exactly is a receiver and a donor?

Imagine one set of **ten images following R1**, and another set of **ten images following R2**.

The **receiver** is the input whose internal calculation we deliberately change. The **donor** is the input supplying the replacement activation.

We first ask Qwen to process both image sets normally, and we record their answer probabilities. Then, while Qwen processes the receiver again, we replace **one activation vector from one decoder layer** with the vector from the donor at the **same layer and token position**.

We let the rest of the receiver's calculation continue and see how its final rule probabilities change.

The images and prompt stay the same during the patched receiver run. The patch changes a temporary internal value, **not the saved LoRA or pretrained weights**.

A decoder layer is a stage of processing in Qwen's language model. This model has **32 decoder layers**. We can patch one chosen layer at a time.

There are two types of receiver–donor pairs:

| Pair type | Example | Why we need it |
|---|---|---|
| **Same-rule** | R1 receiver and a different R1 donor | Control: changing vectors without changing the intended rule can still affect the answer |
| **Different-rule** | R1 receiver and R2 donor | Test: does the receiver become more likely to select R2? |

A **donor's true rule** is the rule used to select its images. A **donor's predicted rule** is what Qwen actually answered. They may differ. If Qwen copies the donor's wrong prediction, that is **not** evidence of correct rule transfer.

## 3. Why can't we simply compare with the old Experiment 03 CSV?

The completed Experiment 03 used its own selection of held-out images and 3,456 patching trials. For LoRA, we have now defined a new **800 training / 100 validation / 100 test image split**, with newly constructed ten-image evaluation sets.

If we compared an old pretrained run on one selection of images against a fine-tuned run on another, an apparent change might be due to **different questions** rather than fine-tuning.

So this experiment requires a **fresh pretrained patching run on the new evaluation sets**, followed by a LoRA patching run on **exactly the same pairs**.

The old Experiment 03 results and code are preserved; we do not overwrite them.

The comparison script checks that the two new runs agree on the same dataset manifest, receiver–donor plan, layers, token positions, rule-to-letter mappings, base-model revision and image preprocessing. If they do not, the comparison fails instead of silently combining unrelated results.

Use the **validation** sets while deciding which layers or settings to investigate. Use the **final test** sets only after those choices are fixed.

## 4. How do we choose the receiver and donor sets?

We have **30 validation ten-image sets** in Experiment 04, ten for each of R1, R2 and R3. The default patching plan deliberately starts smaller, because every pair requires many model runs.

With `--receivers-per-rule 1`, we choose:

- One R1 receiver.
- One R2 receiver.
- One R3 receiver.

That gives **3 receiver sets**.

For each receiver, the program selects one donor for each rule, including a **different set with the same rule**. Thus each receiver gets three donors: one same-rule and two different-rule.

The result is:

| Quantity | Count |
|---|---:|
| Receivers | 3 (one per rule) |
| Donors selected for each receiver | 3 |
| Receiver–donor pairs | **9** |
| Same-rule pairs | 3 |
| Different-rule pairs | 6 |

The selector tries to use donors whose image filenames **do not overlap** the receiver's. If none are available, it picks the candidate with the **fewest shared image files** and records the overlap. A set is never paired with itself.

This matters because our validation and test partitions each contain only **100 distinct pictures**, reused across their sets. The pairs are not automatically image-disjoint just because they have different set IDs.

You can change the receiver count later with `--receivers-per-rule`, but use **exactly the same value and resulting pair plan** for the original and LoRA models.

## 5. What are we changing in each patching trial?

A **trial** means one receiver–donor pair, one answer-option rotation, one patched layer and one token position.

### Which layer?

The default small plan patches layers:

`1, 8, 16, 24, 32`

This samples early, middle and late processing stages without immediately testing every layer. It does **not** claim these are the scientifically important layers.

You may later pass other layer numbers using `--layers`. A full 32-layer sweep is also possible but more expensive.

### Which token position?

We perform two separate patches:

- **Final token**: the last token position in the *prepared prompt* before the answer is predicted. Its representation is read to score X, Y or Z.
- **Previous token**: the token immediately before that. This is a control for how much the effect depends on patching the final position.

These are **not the last two pictures**, and they are **not two generated answer letters**.

At the last decoder layer, patching the final-token vector can change the answer directly before the final readout. A large effect there may reflect answer copying rather than where the logical rule was computed.

### Why rotate X, Y and Z?

The same three rules are shown with three different cyclic mappings:

| Rotation | X | Y | Z |
|---|---|---|---|
| 0 | R1 | R2 | R3 |
| 1 | R2 | R3 | R1 |
| 2 | R3 | R1 | R2 |

Both donor and receiver use the **same mapping in one patching trial**. This makes it easier to notice if the model is simply favouring a certain answer letter.

Three cyclic rotations are **not all six possible permutations**.

## 6. How many trials will we run?

The default selected-layer run has:

| Factor | Count |
|---|---:|
| Receiver–donor pairs | 9 |
| Answer-option rotations | 3 |
| Selected layers | 5 |
| Token positions | 2 |

So:

~~~text
9 pairs × 3 rotations × 5 layers × 2 positions
= 270 patching trials PER MODEL.
~~~

We do this on **the pretrained model and on the LoRA model**:

~~~text
270 original-model patching trials
270 LoRA-model patching trials
--------------------------------
540 patching trials in the matched comparison
~~~

These are not 540 different image sets. The same sets and pairs are reused at different layers, token positions and answer orders.

For the **different-rule** condition alone, we have six pairs × three rotations = **18 different-rule trials at each selected layer and token position**, per model. The **same-rule** condition has three pairs × three rotations = **9 trials** there.

For the default plan, results are therefore averages over **18 different-rule** or **9 same-rule trials**, not the 36 and 18 from historical Experiment 03.

The **smoke test** is smaller: one same-rule pair and one different-rule pair, three rotations, layers 1 and 32, two positions. That gives **24 patching trials per model**. The program also makes normal forward passes to record donor and receiver vectors; those are extra computation, not additional patching rows.

If we test all 32 layers with the same nine pairs, the count becomes:

`9 × 3 × 32 × 2 = 1,728 trials per model.`

## 7. How does the code perform activation patching?

The script is [`patch_finetuned.py`](../../code/05-post-lora-comparison/patch_finetuned.py).

For each pair and answer rotation, it:

1. Builds the original ten-image receiver prompt and asks Qwen for its normal prediction.
2. Saves the receiver's probabilities and the internal vectors at decoder layers 1–32.
3. Does the same for the donor.
4. Selects the requested layer and either the final or previous prompt-token position.
5. Runs the receiver again and uses a **forward hook** to swap that one vector with the donor's corresponding vector.
6. Recalculates the receiver's X/Y/Z probabilities and predicted rule.
7. Saves the original, donor and patched results with the pair, layer, rotation and token position.

A **forward hook** is just a callback run when a model layer produces an output. The code replaces one token vector in that output, lets later computations continue, and removes the hook after scoring. It does not replace every layer or all prompt tokens.

The same script works with either model:

- Without `--adapter`: run the original pretrained Qwen.
- With `--adapter "$ADAPTER"`: load the same base Qwen **plus the trained adapter**.

This is how we avoid writing two different patching methods that might behave differently for unrelated reasons.

## 8. What exactly do we measure?

Suppose a receiver's true rule is R1 and the donor's true rule is R2.

Before patching, imagine the receiver assigns R2 a probability of 20%. After patching, it assigns R2 35%.

The change toward the donor's **true rule** is:

~~~text
35% - 20% = +15 percentage points.
~~~

If it falls from 20% to 12%, the signed change is **−8 percentage points**.

The program measures several related but different things:

| Measurement | The question it answers |
|---|---|
| **Signed donor-rule probability change** | Does the donor's actual generating rule become more (+) or less (−) likely, on average? |
| **Absolute donor-rule probability change** | How big are the shifts, ignoring whether they are positive or negative? |
| **Receiver-rule probability change** | Does the receiver gain or lose confidence in its own true rule? |
| **Answer-change rate** | How often does the top predicted rule switch? |
| **Changed to donor's true rule** | How often does the patched answer switch to the donor's generating rule? |
| **Changed to donor's predicted rule** | How often does it switch to what the donor *answered*, even if that answer was wrong? |

**Why do we need both signed and absolute changes?** Consider two trials with changes of +10 and −10 percentage points. Their signed mean is **zero**, but their mean absolute change is **10 percentage points**. A zero signed average does not mean patching did nothing.

**Why do we need same-rule controls?** Even if the donor and receiver have the same true rule, their images, ordering and representations differ. An effect that also appears in same-rule pairs is not automatically evidence of rule-specific transfer.

**Why distinguish donor truth from donor prediction?** A patch might simply copy a mistaken donor answer. That is an effect on predictions, but not correct transfer of the generating rule.

These metrics tell us **whether and how much outputs changed**. By themselves, they do **not** establish how Boolean logic was computed internally.

## 9. How do we compare pretrained and LoRA patching results?

The comparison program is [`compare_patching.py`](../../code/05-post-lora-comparison/compare_patching.py).

It reads two saved, completed runs without loading Qwen. It first checks that both runs used **matching examples and settings**.

It then calculates the above metrics for each:

~~~text
Condition (same / different)
         ×
Token position (final / previous)
         ×
Decoder layer
~~~

For each combination, it saves a **before** value, an **after** value, and their difference:

~~~text
Change caused by fine-tuning in a patching metric
     = LoRA-model metric - pretrained-model metric
~~~

This is a *difference between two measured effects*. It is **not itself an activation patch**, and it is **not the probability shift from one individual trial**.

We should look for patterns such as:

- Did held-out rule-prediction accuracy improve after LoRA?
- Did different-rule patching become more likely to increase the donor's **true-rule** probability?
- Did that change exceed what also happened with **same-rule controls**?
- Did answer-changing behaviour become stronger only at the very last layer, or elsewhere too?
- Did the model become less biased toward particular answers or rules?

The code computes **descriptive comparisons**. It does not run a statistical test proving that a change is significant, and it does not discover a uniquely correct reasoning layer.

## 10. Where are the files and graphs?

Each patching run saves:

| File | Meaning |
|---|---|
| `config.json` | Model revision, precision, image settings, adapter identity and complete plan |
| `manifest.json` | Exact dataset split and ten-image sets |
| `pair_plan.json` | Exact receiver–donor pairs, their true rules, shared images, rotations, layers and positions |
| `patches.jsonl` | Every patching trial: original receiver/donor and patched answer probabilities |
| `complete.json` | Confirms the expected patching trials were recorded |

The comparison saves:

| File | Meaning |
|---|---|
| `before.json` | Summary of pretrained-model patching |
| `after.json` | Summary of LoRA-model patching |
| `differences.csv` | For each layer/condition/position, LoRA summary minus pretrained summary |
| `sources.json` | Which original runs and settings the comparison used |
| `signed_donor.png` | Change toward/away from the donor's true rule |
| `absolute_donor.png` | Magnitude of donor-rule probability shifts |
| `receiver_change.png` | Change in probability of the receiver's true rule |
| `answer_change.png` | Rate at which the chosen answer changes |
| `donor_true_transfer.png` | Switches to the donor's actual rule |
| `donor_prediction_transfer.png` | Switches to the donor's predicted answer |

Read a graph by checking: **which model, same/different rule, final/previous token, and which layer?** A larger curve does not automatically mean better reasoning.

The comparison uses **all selected trials**, not just cases where the original model or the adapter answered correctly. It refuses incomplete or mismatched runs.

## 11. How to run Experiment 05 in Colab

First complete the setup and fine-tuning in the [Experiment 04 guide](../04-lora/README.md#9-how-to-run-experiment-04-in-colab).

You need the same exported variables: `CODE`, `DATA`, `MANIFEST`, `REV`, `PATCH_RUNS` and the saved `ADAPTER` folder. In Colab, a new `%%bash` cell **does not remember** exported variables from previous shell cells, so redefine them or run the commands together.

### 11.1 Check the pair plan on CPU

~~~bash
python "$CODE/05-post-lora-comparison/patch_finetuned.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$PATCH_RUNS/pretrained" \
  --partition validation --validate-only
~~~

This prints how many receiver–donor pairs and patching trials are planned, and reports how many pairs share images. The `--validate-only` mode does **not** load the large model.

### 11.2 Try the small real GPU runs

~~~bash
python "$CODE/05-post-lora-comparison/patch_finetuned.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$PATCH_RUNS/pretrained" \
  --partition validation --smoke-test

python "$CODE/05-post-lora-comparison/patch_finetuned.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$PATCH_RUNS/finetuned" \
  --adapter "$ADAPTER" --partition validation --smoke-test
~~~

Each smoke test uses **two pairs** and layers **1 and 32**, with both token positions and all three rotations. This is only a runtime check, **not a research result**.

### 11.3 Run the matched selected-layer experiment

~~~bash
python "$CODE/05-post-lora-comparison/patch_finetuned.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$PATCH_RUNS/pretrained" \
  --partition validation --receivers-per-rule 1 \
  --layers 1 8 16 24 32

python "$CODE/05-post-lora-comparison/patch_finetuned.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$PATCH_RUNS/finetuned" \
  --adapter "$ADAPTER" --partition validation \
  --receivers-per-rule 1 --layers 1 8 16 24 32
~~~

Each complete run has **270 patches** at the specified five layers.

The commands print the full, settings-specific result folders. **Copy those printed paths**, because the run directory names are generated from the configuration. Do not guess the folder names.

### 11.4 Compare the saved runs on CPU

~~~bash
python "$CODE/05-post-lora-comparison/compare_patching.py" \
  --baseline /full/pretrained/patching/run \
  --finetuned /full/finetuned/patching/run \
  --output "$PATCH_RUNS/comparison-validation"
~~~

The comparison output directory must be **new and empty**. You can inspect the plots there without loading Qwen or running any patches again.

After the experimental decisions are fixed, repeat the two patching commands with `--partition test`, then compare the two final-test run folders into a **different new output folder**. Do not use test results to decide which layers to patch.

### What if we want all 32 layers?

The script accepts explicit layer numbers. For instance:

~~~bash
--layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
~~~

Use this **same exact list** for pretrained and LoRA runs. All-layer patching is more expensive, so first verify the five-layer version.

## 12. How should we interpret the possible outcomes?

No new GPU results have been collected yet, so the following are **possibilities**, not findings:

**Case A — Accuracy improves, and different-rule patches more often move toward the donor's true rule.**

That would suggest the fine-tuned model behaves differently on the task, and the internal interventions show more rule-directed effects. However, we still need same-rule controls, donor-correctness checks, and larger evaluation sets before claiming a computation is localized.

**Case B — Accuracy improves, but patching effects barely change.**

Then LoRA helped rule prediction without producing an obvious change in this specific single-vector patching measurement. It would **not** prove that the internal computation was unchanged.

**Case C — Large patching changes appear near layer 32, but also in same-rule pairs.**

That can happen if the final representation directly influences the answer readout. It is **not** enough to say Boolean reasoning happens at layer 32.

**Case D — Accuracy and patching results remain poor.**

We would have learned that this training setup did not produce the hoped-for improvement under our measurements. It would not mean VLMs cannot learn Boolean rules in general.

The completed historical Experiment 03 provides a warning: its final-token different-rule **signed means were negative across all 32 layers**, layer 6 was only **least negative**, and layer 32 often copied the donor's **predicted** answer rather than its correct rule. Those are **historical** results on different test sets, not numbers to compare directly with this new run.

## 13. What this experiment can and cannot prove

Activation patching is a **causal intervention**: we deliberately replace a vector and watch whether the model's answer changes. That tells us that this replacement can influence the answer.

It does **not** automatically mean the patched layer computes the logical rule.

Our evidence is limited by:

- **Positive-only image sets:** one decisive picture may reveal the rule.
- **Image reuse:** evaluation pairs may share pictures, and many trials reuse the same sets.
- **Three answer orders:** these help check answer bias, but they are not independent inputs or all possible permutations.
- **Small default patching plan:** 9 pairs give a useful pilot, not a broad statistical sample.
- **No statistical significance test:** the plots describe observed differences; they do not establish unique reasoning locations.
- **Output-position effects:** late final-token patches can directly change the answer readout.

For Monday, we want a **careful before/after comparison**: what changed in rule accuracy, what changed in donor-rule probability shifts, and whether those changes go beyond same-rule controls.

If we later try **single-layer LoRA**, we must first define *how* to choose that layer (for example, a rule-specific effect beyond same-rule controls on validation data). We should not automatically pick the layer with the biggest absolute shift or the least-negative signed mean.

**Where to go next:** read the [Experiment 04 guide](../04-lora/README.md), then inspect `before.json`, `after.json`, `differences.csv` and the comparison graphs **after the GPU runs complete**.
