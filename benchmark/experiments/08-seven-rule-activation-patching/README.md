# 08 — Does Qwen know more than its answer shows?

In [experiment 07](../07-seven-rule-operator-controls/README.md), Qwen struggled to identify the rule behind a set of pictures. It often chose one of the last two options, even when we changed the order of the answers.

But that leaves another question: **does Qwen fail to recognize the rule at all, or does something go wrong when it turns what it has seen into an answer?**

To explore this, we looked inside the model using *activation patching*.

## The idea

Imagine showing Qwen two different sets of pictures. Each set follows a known rule.

We first let the model process them normally. Then we take a small piece of its internal activity while it processes one set (the *donor*) and replace the corresponding piece while it processes the other (the *receiver*).

If the donor's rule is represented in that piece of activity, moving it might make the receiver more likely to choose that rule.

We also swap activity between two sets that follow the **same** rule. This gives us something to compare against: a change in the answer does not automatically mean we transferred information about a different rule.

This is not training. The model's weights stay the same throughout.

## What we tried

We used the pretrained Qwen3.5-4B model and the saved seven-rule validation dataset from experiment 07.

We made 14 donor–receiver pairs: seven with the same rule and seven with different rules. We repeated the swaps at each of the model's 32 layers, checking two positions near the answer. This gave us **896 measurements**.

For each swap, we looked at whether Qwen became more likely to pick the donor's *actual* rule, whether it leaned toward the donor's *own answer* (which could be wrong), and whether its final choice changed.

## What happened

The swaps did affect Qwen's answers, but not in one clear direction. At some layers, the probability of the donor's true rule went up; at others, it went down.

Changes were generally more noticeable toward the later layers. At the final layer, swapping between different-rule examples raised the donor-rule probability by about **0.8 percentage points** on average. However, swaps between examples with the same rule produced a larger shift, about **3 percentage points**.

![How the effect of swapping internal activity changes across layers](../../results/seven-rules/qwen3.5-4b/activation-patching/layer_effects.png)

So we cannot say that we found a specific place where Qwen stores or applies the Boolean rule. The swaps changed its behavior, but the changes could reflect other information, not necessarily an understanding of the rule itself.

This is also a limited experiment: we tested a small selection of pairs, kept the answer order fixed, and some examples share images. A different selection could give a different picture.

You can inspect the [measurements](../../results/seven-rules/qwen3.5-4b/activation-patching/layer_effects.csv) and [individual swaps](../../results/seven-rules/qwen3.5-4b/activation-patching/patches.csv).

## Running the experiment

You need the original image dataset and manifest, plus the verified prediction cache produced by [experiment 07](../07-seven-rule-operator-controls/README.md). The large input files are not stored in Git.

Install the requirements from the repository root and run:

```bash
pip install -r benchmark/code/00-common/requirements-seven-rule-controls.txt
python benchmark/code/08-activation-patching/activation_patching_7rules.py --manifest PATH_TO_MANIFEST --data-dir PATH_TO_DATA --cache PATH_TO_07_OUTPUT --out PATH_TO_OUTPUT --run
```

Replace the paths with local locations. Keep the output folder outside the repository. This step loads the model and needs a compatible GPU setup matching the saved inference environment. Without `--run`, the script only shows the planned work. For reproducing or resuming the published pre-reorganization run, use its [matching original code](../../code/README.md#earlier-saved-runs), since the moved scripts have different source hashes.

**What this tells us:** Changing the model's internal activity can change its answers, but that alone does not show that it has learned to apply logical rules. We still need stronger tests to separate actual rule information from answer preferences and other effects.
