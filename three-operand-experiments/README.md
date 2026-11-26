# Three-rule experiments

The model sees images with three visual concepts. It chooses the rule that generated the set. The first experiments used 48 images and three candidate rules.

| Experiment | Question | What I found |
|---|---|---|
| [01 · Operator controls](experiments/01-operator-controls/README.md) | Can Qwen recognize the objects and find the hidden rule? | Both models recognized the concepts in all eight control images. Main rule accuracy was close to 33.3% chance. |
| [02 · Probing](experiments/02-probing/README.md) | Can a classifier recover the rule from Qwen’s internal states? | In the later Qwen3.5 run, the final-layer probe scored 100% with one selected image and 50% with ten. Scene cues could help explain this. [Saved probe results](results/02-probing/qwen3.5-4b/README.md) |
| [03 · Activation patching](experiments/03-activation-patching/README.md) | Does swapping a hidden state change the answer? | Across 3,456 interventions, no layer met the study’s rule-transfer criterion. |
| [04 · LoRA](results/04-lora/qwen3.5-4b/README.md) | Does fine-tuning improve rule selection? | Accuracy went from 29/90 to 30/90. The selected adapter always answered Y. |

[Saved results and exploratory controls](results/README.md) include the 10-versus-20-image pilot and the three-versus-seven answer-choice control. The next benchmark used all seven rules as generators. [Seven-rule benchmark](../benchmark/README.md)
