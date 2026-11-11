# 06 — Does fine-tuning help Qwen identify seven Boolean rules?

The pretrained model struggled to identify the rule connecting ten pictures. In [experiment 07](../07-seven-rule-operator-controls/README.md), it got only **48 of 343** validation questions right and strongly favored the last two answer positions. We wanted to see whether a small amount of additional training could change this.

We used **LoRA** (low-rank adaptation): instead of changing the entire Qwen3.5-4B model, it trains a small set of extra weights. The question is not just whether its validation accuracy increases, but **whether it starts to distinguish rules rather than repeat the same answer bias**.

## What we tried

We retained the saved seven-rule dataset and its train/validation/test split. The model learns from **147 training sets**; **49 validation sets** are used for checkpoint selection and evaluation. We keep the test partition untouched.

The planned full run uses three epochs, gradient accumulation over four training sets and **111 successful optimizer updates**. It uses Qwen3.5-4B with NF4 four-bit base weights, FP16 computation, and LoRA rank 8, alpha 16, dropout 0.05. AdamW has learning rate 5×10⁻⁵, betas (0.9, 0.999), epsilon 10⁻⁸ and weight decay 0.01.

The selected checkpoint is the one with the lowest validation **answer-token loss** at epoch boundaries—not necessarily the one with the highest validation accuracy. This matters: it prevents us from choosing a model just because it happens to look best on the reported accuracy metric.

After training, the evaluation should compare the selected LoRA model and pretrained Qwen on the **same validation examples and answer rotations**.

## Results

The verified pretrained result is **48/343 (13.99%)**, close to the seven-choice chance level of **14.29%**.

The older files currently visible on GitHub contain an interrupted training attempt with **one verified successful optimizer update**, not a completed full run. A separate historical notebook mention of **116/343** is not sufficient evidence for a final LoRA score.

**The latest completed-run archive has not yet been verified and imported here.** We therefore do not report a fine-tuned accuracy, best checkpoint or training curve until the original completion record, selection, losses, final metrics and per-example predictions agree.

The [LoRA results folder](../../results/seven-rules/qwen3.5-4b/lora/README.md) will show the measured before/after results and original plots when they are verified. The pretrained individual predictions and answer-position analysis are already in [operator controls](../../results/seven-rules/qwen3.5-4b/operator-controls/README.md).

## What would count as improvement?

We would look for a higher fine-tuned validation accuracy **and** less reliance on fixed answer positions. The per-rule comparison matters as well: a score can improve while the model continues to predict only one or two rules.

Even a genuine gain in accuracy would not by itself prove that Qwen learned to apply Boolean logic. The validation examples reuse images across sets, and rotations of the same set are correlated. The held-out test partition has not been evaluated.

## Code

The implementation is [`benchmark/code/06-lora/lora_7rules.py`](../../code/06-lora/lora_7rules.py), with a short [code guide](../../code/06-lora/README.md). The original data manifest, model revision, package versions and source-code hashes must match for saved checkpoints to be reproduced or resumed; see [compatibility notes](../../code/README.md#earlier-saved-runs).

The experimental findings are recorded in the results folder rather than in checkpoint or backup instructions.
