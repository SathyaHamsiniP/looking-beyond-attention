# Looking Beyond Attention

We test whether vision-language models can find a Boolean rule from a set of images.

A is a green cube. B is a blue sphere. C is a red cylinder. We give the model ten images and ask which rule explains them.

Earlier experiments compared Qwen3-VL-4B and Qwen3.5-4B. New runs use **Qwen3.5-4B only**.

## Three versions of the task

V1 asks which rule explains ten images that all satisfy it. The three rules are R1 = `(A AND B) OR C`, R2 = `(A AND C) OR B`, and R3 = `(B AND C) OR A`.

There is a shortcut. Bits describe whether A, B and C are present. A positive `001` identifies R1, `010` identifies R2, and `100` identifies R3. One image can therefore reveal the answer without testing the whole expression.

V2 shows both true and false examples. For example, a false `010` rules out R2 but leaves R1 and R3. A false `100` then rules out R3. Together they identify R1, although neither observation does so alone. The generator checks this property for every set.

V3 adds eight different Boolean rules, including XOR, negation and different groupings of AND and OR. This lets us test more than three closely related expressions. V2 and V3 each use five true and five false observations per ten-image set. See [the task guide](three-operand-experiments/tasks/README.md) for the exact rules and checks.

Dataset-v1 originally has 800 training and 200 held-out images. The newer pipeline divides the held-out group into 100 validation images for choosing settings and 100 final-test images for final evaluation. All groups contain the same Boolean configurations and rules, but different image files. This does not test whether the model can handle new kinds of logical expressions.

## Experiments

- [01 — Operator controls](three-operand-experiments/experiments/01-operator-controls/README.md): Ask the model to find the rule. Test changes to prompts and answer choices.
- [02 — Probing](three-operand-experiments/experiments/02-probing/README.md): Check which object and rule information is present at each model layer.
- [03 — Activation patching](three-operand-experiments/experiments/03-activation-patching/README.md): Same-rule and different-rule patching in Qwen3.5-4B. The full baseline run is complete.

- [04 — LoRA](three-operand-experiments/experiments/04-lora/README.md): Train small adapters while keeping the base model frozen, then check accuracy on held-out images.
- [05 — Post-LoRA patching](three-operand-experiments/experiments/05-post-lora-comparison/README.md): Repeat matched activation changes before and after training.

The earlier experiments used **48 images**. Their saved results have not been changed.

The Qwen3.5-4B baseline patching run on dataset-v1 is complete. Read [Experiment 03](three-operand-experiments/experiments/03-activation-patching/README.md) for the beginner guide, reported findings, verification status and results location.

## Get the 1,000 images

Download the [dataset-v1 release](https://github.com/SathyaHamsiniP/looking-beyond-attention/releases/tag/dataset-v1), or run:

```bash
python three-operand-experiments/code/00-data-generation/download_dataset.py
```

This creates `three-operand-experiments/data-1000/` with:

- `images/` — 1,000 images.
- `metadata.json` — object labels and train/test groups.

There are 800 training images and 200 held-out images. The download is checked before it is saved.

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


Follow [COLAB.md](three-operand-experiments/COLAB.md) for exact commands, smoke tests and results folders. [IMPLEMENTATION.md](three-operand-experiments/IMPLEMENTATION.md) lists what was built and tested. New LoRA and V3 model runs still need GPU testing; no LoRA or V3 model results are claimed.
