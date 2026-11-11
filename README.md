# Looking Beyond Attention

Can vision-language models identify the Boolean rule shared by a set of images? We study this through model answers, probes of internal representations, activation patching and LoRA fine-tuning.

A is a green cube, B is a blue sphere and C is a red cylinder. For example, `(A AND B) OR C` is true when an image contains both A and B, or contains C. The model sees positive examples and chooses a rule from a list.

We started with three rules, then extended the task to seven. Earlier experiments used Qwen3-VL-4B and Qwen3.5-4B; the newer comparisons use **Qwen3.5-4B**. See the [benchmark overview](benchmark/README.md) for methods, results and instructions, with three-rule and seven-rule work kept separate.

## Experiments

- [01 — Operator controls](three-operand-experiments/experiments/01-operator-controls/README.md): How do prompts, answer choices and image counts affect rule identification?
- [02 — Probing](three-operand-experiments/experiments/02-probing/README.md): Which object and rule information can be decoded from model layers?
- [03 — Activation patching](three-operand-experiments/experiments/03-activation-patching/README.md): How do answers change when we replace an internal representation with one from another example? Saved results are available.
- [04 — LoRA](benchmark/results/three-rules/qwen3.5-4b/lora/README.md): Three-rule validation accuracy changed from 29/90 to 30/90; the selected adapter always answered Y.

The [newer experiments](benchmark/README.md) include completed seven-rule operator controls and activation patching. Seven-rule probing is implemented but has not run on GPU. Seven-rule LoRA has partial training records, with no verified full-run adapter or completed fine-tuned evaluation. These studies differ in data and method, so they are not perfectly matched comparisons.

## Data

The original 48-image experiments remain available. Later work uses the [1,000-image dataset](https://github.com/SathyaHamsiniP/looking-beyond-attention/releases/tag/dataset-v1):

```bash
python three-operand-experiments/code/00-data-generation/download_dataset.py
```

This creates `three-operand-experiments/data-1000/` with `images/` and `metadata.json`. The release has 800 training and 200 test images. The seven-rule experiments use a saved 800/100/100 train/validation/test assignment and saved ten-image sets; reuse that manifest rather than generating replacements. Images can appear in multiple sets within a partition.

[Image-generation instructions](three-operand-experiments/code/00-data-generation/README.md) describe how the dataset was made.

## Results and how to run

The [result index](benchmark/results/README.md) links saved measurements, plots and settings. Original experiment files and result locations remain unchanged. Dataset images, large activation arrays and adapters are stored outside GitHub.

The experiment READMEs explain how to run each method. New outputs are saved locally in Colab, printed in the console and then backed up to the School Drive account. See [seven-rule probing](benchmark/experiments/09-seven-rule-probing/README.md) for setup and [LoRA](benchmark/experiments/06-seven-rule-lora/README.md) for checkpoint recovery. Older runners still accept their original configurable data and result paths.

## Credit

This work builds on [Roshan's two-operand experiments](https://github.com/RoshanYSingh23/looking-beyond-attention). Our image generator and Qwen3-VL runs use parts of that public repository. Qwen3.5 uses a separate model loader.
