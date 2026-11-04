# 04 — LoRA fine-tuning

## What we want to find out

Can a small amount of training help Qwen3.5-4B choose the right rule? We give it ten images, their true/false labels, and a list of possible rules.

LoRA adds small trainable adapters while keeping the original model weights fixed. QLoRA also loads the base weights in 4-bit form to save memory. We either train adapters across decoder layers or start a separate adapter at one selected layer. Each run starts from the same pretrained base.

The code checks which modules exist in each layer and records exactly which ones it trains. It changes the option order during training so the correct answer is not always the same letter. We use validation loss to choose a checkpoint, then measure model accuracy, per-rule accuracy and consistency across answer orders on held-out examples.

## Data and commands

Use dataset-v1 V3 balanced ten-image labelled sets. The original 800 training images supply optimizer examples. Derived 100-image validation sets select the checkpoint; 100 final-test images are reserved. See the executable setup, pretrained baseline prerequisite, training, resume and evaluation commands in [COLAB.md](../../COLAB.md#04-qlora).

## Outputs

Configuration, split/task manifests, model revision, trainable module/parameter list, per-step training and validation losses, optimizer checkpoints, `best_adapter`, selection criterion, and completion marker. Evaluation saves per-example probabilities, per-rule and overall accuracy, chance/majority baselines, order consistency and ambiguity-aware shortcut controls.

## Results and limitations

No pretrained-model LoRA run has been executed here. CPU tests use tiny random-weight Qwen3.5 models and real PEFT to test freezing, selected targets, forwards/backwards and adapter reloads. Real 4-bit loading, training resumption and ten-image T4 memory need GPU smoke validation. Rank, resolution, target subset and accumulation are configurable. Accuracy on shared rule structures does not establish compositional generalization or logical reasoning.
