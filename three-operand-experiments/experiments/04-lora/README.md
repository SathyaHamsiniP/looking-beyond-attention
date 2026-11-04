# 04 — LoRA fine-tuning

## Goal and method

Test whether training small adapters improves Qwen3.5-4B rule selection. QLoRA freezes the pretrained model and trains low-rank updates to decoder linear modules. Across-layer and fresh single-layer runs use the same training manifest. The actual module list and trainable counts are saved, rather than assuming every decoder layer uses the same attention architecture.

## Data and commands

Use dataset-v1 V3 balanced ten-image labelled sets. The original 800 training images supply optimizer examples. Derived 100-image validation sets select the checkpoint; 100 final-test images are reserved. See the executable setup, untuned baseline prerequisite, training, resume and evaluation commands in [COLAB.md](../../COLAB.md#04-qlora).

## Outputs

Configuration, split/task manifests, model revision, trainable module/parameter list, per-step training and validation losses, optimizer checkpoints, `best_adapter`, selection criterion, and completion marker. Evaluation saves per-example probabilities, per-rule and overall accuracy, chance/majority baselines, order consistency and ambiguity-aware shortcut controls.

## Results and limitations

No pretrained-model LoRA run has been executed here. CPU tests use tiny random-weight Qwen3.5 models and real PEFT to test freezing, selected targets, forwards/backwards and adapter reloads. Real 4-bit loading, training resumption and ten-image T4 memory need GPU smoke validation. Rank, resolution, target subset and accumulation are configurable. Accuracy on shared rule structures does not establish compositional generalization or logical reasoning.
