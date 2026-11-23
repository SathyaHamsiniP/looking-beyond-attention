# Investigating Boolean rule induction in vision-language models

We asked whether Qwen could infer a rule from positive visual examples. A green cube, a blue sphere and a red cylinder stood for three Boolean variables. Different combinations made it possible to test the same rule across many scenes.

## Can Qwen find the hidden rule?

Small three-rule controls showed that Qwen could recognize objects and evaluate a given expression. When asked to choose the hidden rule, its answer often followed the option position. In the seven-rule task we rotated the options for each of 49 sets. Pretrained Qwen3.5-4B scored 48/343, close to the one-in-seven chance rate. The suites used different image sets, so comparing their scores cannot isolate the effect of adding rules. [Three-rule controls](three-operand-experiments/experiments/01-operator-controls/README.md) · [Seven-rule controls](benchmark/experiments/07-seven-rule-operator-controls/README.md)

## What can a classifier read from Qwen?

We trained separate classifiers on activity inside the pretrained model. The seven-rule probe used a fixed answer order. Each column below reports correct answers across the same 49 validation sets.

| Images per set | Qwen | Final-layer probe | Probe trained with shuffled labels |
|---:|---:|---:|---:|
| 1 | 7 | 15 | 5 |
| 5 | 7 | 19 | 5 |
| 10 | 7 | 10 | 8 |

A scene can fit several rules and still occur more often with one label. That could help explain the one-image score. The five-image selection also used the generating rule. Some images appeared in several sets. These details leave open whether the classifier read a visual shortcut or evidence combined across scenes. [Probe results and layer scores](benchmark/results/seven-rules/qwen3.5-4b/probing/README.md)

## What did LoRA change?

The early three-rule run moved from 29/90 to 30/90 correct. Its selected adapter always chose Y. For seven rules we trained on 147 sets over three epochs and recorded 111 successful updates. On the matched validation prompts, accuracy rose from 48/343 to 112/343. R4 went from 12 to 38 correct and R5 from 1 to 31, each out of 49. R1 and R7 got worse. Qwen chose the same rule across all seven orders on eight sets after training and zero before. A consistent choice can still be wrong.

Validation loss selected the checkpoint. An independent test on fresh scenes would tell us more about generalization. [Three-rule LoRA](three-operand-experiments/results/04-lora/qwen3.5-4b/README.md) · [Seven-rule training and matched predictions](benchmark/experiments/06-seven-rule-lora/README.md) · [Rebuild the comparison figure](docs/plot_lora.py)

## What happens when we swap internal activity?

In the three-rule baseline, none of 32 layers met the rule-transfer criterion across 3,456 interventions. Copying the final state could copy the donor’s answer. That state sits directly before the answer is read out. Across 896 seven-rule interventions, the final-layer shift toward the donor rule averaged about 0.8 percentage points for different-rule pairs. Same-rule pairs shifted about 3 points. The swaps influenced predictions. Their direction gives weak support for a rule-specific effect. [Three-rule analysis](three-operand-experiments/results/03-activation_patching/README.md) · [Seven-rule analysis](benchmark/experiments/08-seven-rule-activation-patching/README.md)

## What would distinguish the explanations?

Answer-position bias, visual shortcuts and difficulty combining examples could all contribute. On fresh scenes, we would replace one image so the correct rule changes. We would reuse that image with different surrounding examples and rotate the options. A model that follows the evidence should switch to the specific new rule and keep that choice when only its position changes. The separate post-LoRA layer study is in progress. [Experiment design](benchmark/experiments/10-layer-patching/README.md)
