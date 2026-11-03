# Looking Beyond Attention

We test whether vision-language models can find a Boolean rule from a set of images.

A is a green cube. B is a blue sphere. C is a red cylinder. We give the model ten images and ask which rule explains them.

Earlier experiments compared Qwen3-VL-4B and Qwen3.5-4B. New runs use **Qwen3.5-4B only**.

## Experiments

- [01 — Operator controls](three-operand-experiments/experiments/01-operator-controls/README.md): Ask the model to find the rule. Test changes to prompts and answer choices.
- [02 — Probing](three-operand-experiments/experiments/02-probing/README.md): Check which object and rule information is present at each model layer.
- [03 — Activation patching](three-operand-experiments/experiments/03-activation-patching/README.md): Same-rule and different-rule patching in Qwen3.5-4B. Code is ready for the first Colab test; no results yet.

The earlier experiments used **48 images**. Their saved results have not been changed.

## Get the 1,000 images

Download the [dataset-v1 release](https://github.com/SathyaHamsiniP/looking-beyond-attention/releases/tag/dataset-v1), or run:

```bash
python three-operand-experiments/code/00-data-generation/download_dataset.py
```

This creates `three-operand-experiments/data-1000/` with:

- `images/` — 1,000 images.
- `metadata.json` — object labels and train/test groups.

There are 800 training images and 200 test images. The download is checked before it is saved.

To use these images, pass `--data-dir three-operand-experiments/data-1000` to an experiment script. The original 48-image dataset still works with the old `data/` path.

[Generate the images yourself](three-operand-experiments/code/00-data-generation/README.md) if you prefer.

## Where results go

Experiment results are saved to Google Drive. They are not added to GitHub automatically.

Default results folder in Colab:

```text
/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results
```

Use `--experiment-dir` if your results folder is elsewhere. Runs using the 1,000 images save in separate `dataset-v1/` folders, so they do not overwrite the earlier results.

## Credit

This work builds on [Roshan's two-operand experiments](https://github.com/RoshanYSingh23/looking-beyond-attention). Our image generator and Qwen3-VL runs use parts of that public repository. Qwen3.5 uses a separate model loader.
