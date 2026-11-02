# 02 — Probing

## Goal

Check what the model knows after it sees the images.

A **probe** is a small classifier. It reads the model's saved activations (numbers inside the model). It does not change the model.

We test two questions:
- **Concept probe:** Can we tell which of A, B and C are in one image?
- **Rule probe:** Can we tell which rule (R1, R2 or R3) made a set of ten images?

A = green cube, B = blue sphere, C = red cylinder.

## Data

These tests use the **original 48 images**, not the new 1,000-image dataset.

| Test | Training | Testing |
|---|---:|---:|
| Concepts | 32 images | 16 different images |
| Rules | 24 sets of 10 images | 3 sets of 10 images |

The rule sets reuse images within training and within testing. No image appears in both groups. There is only **one test set per rule**, so the rule test is very small.

## Method

1. Show the image or ten-image set to the model.
2. Save the activations after each model layer.
3. Train a small classifier using only the training group.
4. Check its predictions on the test group.

We check three ways of reading activations: image positions, text positions and the final position. We also train a probe with shuffled labels as a check.

The concept prompt is: "Describe the objects in this image."

The rule prompt asks for the rule behind ten images. It does not show the answer choices.

## Results

Concept probing has run on both models. These are the **best layer scores**, not averages.

| Activations used | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Image positions | 68.75% | 87.50% |
| Text positions | 62.50% | 43.75% |
| Final position | 43.75% | 43.75% |

These scores were reported from the Colab runs. The saved results have not been checked again here.

The models contain useful information about the objects. This does **not** mean they can work out the rule.

The concept test has only 16 test images. One wrong prediction changes the score by 6.25 percentage points. Picking the best layer on this small test can also make the result look better than it is.

**Rule probing:** The code is ready, but no GPU results have been reported yet. The test has only three sets, so its accuracy cannot support a strong conclusion. A probe may also identify a rule just by spotting one special type of image, without doing the full reasoning task.

## Run on Colab

Use a GPU. Mount Google Drive. Clone this repository if needed:

```bash
git clone https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
```

The commands below use:

```bash
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data'
export RESULTS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results'
```

**Qwen3.5-4B** (use a fresh Colab session):

```bash
pip install -r three-operand-experiments/code/requirements-qwen35.txt matplotlib
python three-operand-experiments/code/02-probing/probe_concepts.py --model qwen3_5 --data-dir "$DATA" --experiment-dir "$RESULTS" --precision 4bit
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3_5 --data-dir "$DATA" --experiment-dir "$RESULTS" --precision 4bit
```

**Qwen3-VL-4B** (use a different fresh session):

First clone [Roshan's reference code](https://github.com/RoshanYSingh23/looking-beyond-attention) to `/content/roshan/looking-beyond-attention-main`.

```bash
pip install -r three-operand-experiments/code/requirements-qwen3vl.txt matplotlib
export UP='/content/roshan/looking-beyond-attention-main'
python three-operand-experiments/code/02-probing/probe_concepts.py --model qwen3 --upstream "$UP" --data-dir "$DATA" --experiment-dir "$RESULTS" --precision 4bit
python three-operand-experiments/code/02-probing/probe_rules.py --model qwen3 --upstream "$UP" --data-dir "$DATA" --experiment-dir "$RESULTS" --precision 4bit
```

Before a full run, you can add `--validate-only` to check the data or `--smoke-test` for a short test. Run one command at a time. Restart the session before switching models.

## Saved files

Results go to Google Drive, under `results/probing/<model>/concept/` or `results/probing/<model>/rule/`.

- `.npz`: saved activations.
- `.csv`: predictions and layer scores.
- `.png`: plots of scores across layers.
- `config.json`: settings used for the run.

Run the same command again to reuse completed files after a disconnect.

The probing scripts still assume the old 48-image data. They must be updated before using the 1,000 new images.
