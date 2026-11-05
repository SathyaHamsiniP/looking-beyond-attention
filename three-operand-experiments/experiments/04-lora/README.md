# 04 — Teaching Qwen to identify Boolean rules with LoRA

## 1. What question are we trying to answer?

In [Experiment 03](../03-activation-patching/README.md), we showed Qwen3.5-4B ten pictures and asked it to identify the Boolean rule that explained them. We then replaced one internal activation vector at a time to see how its answer changed.

Now we want to ask two new questions:

1. **If we give Qwen a little training on this exact task, will it identify the correct rules more often?**
2. **If it improves, will activation patching affect it differently afterward?**

We must compare the model **before and after training on the same evaluation examples**. Otherwise, a difference in performance might come from changing the questions rather than changing the model.

The complete plan is:

~~~text
Existing 1,000 images
        |
        v
Build fixed ten-image sets
        |
        +----> Pretrained Qwen: save its answers
        |
        +----> Train small LoRA adapters on 90 sets
                         |
                         v
                  Qwen + trained adapters
                         |
                         v
                 Save answers on the SAME sets
                         |
                         v
               Compare answers and accuracy
                         |
                         v
          Repeat matched activation patching (Exp 05)
~~~

**Status:** The code is implemented, but this experiment has **not yet been run successfully on the full pretrained model**. There are no real LoRA performance results to interpret yet. CPU tests do not prove that ten-image QLoRA training will fit on a Colab T4.

## 2. What exactly is the task?

An image can contain three important objects:

- **A**: green cube present.
- **B**: blue sphere present.
- **C**: red cylinder present.

Other objects are distractions. A Boolean rule is a statement about which combinations of A, B and C count as true.

| Name | Rule |
|---|---|
| R1 | `(A AND B) OR C` |
| R2 | `(A AND C) OR B` |
| R3 | `(B AND C) OR A` |

For example, `001` means A is absent, B is absent and C is present. That configuration satisfies R1, but not R2 or R3.

**One training example is an entire set of ten actual images, not a single image.** All ten images satisfy one of the rules. Qwen sees the pictures and a text prompt listing the three possible rules; its job is to answer **X, Y or Z**.

For example, imagine the prompt shows:

~~~text
X: (A AND B) OR C
Y: (A AND C) OR B
Z: (B AND C) OR A
~~~

If the ten images were selected using R2, the correct training answer is **Y**. For another set, the rule order might be different, so the correct letter must change too.

We reuse the exact original image-plus-text prompt from [Experiment 03](../03-activation-patching/README.md#5-what-exactly-does-qwen-receive). It tells Qwen that every picture follows one hidden rule, defines A/B/C, lists the choices and asks for one answer label. The actual images are sent to the model; **we do not give it the ground-truth A/B/C bit values**. Those bits are only for preparing and checking the dataset.

### A limitation built into our original task

Our task is **positive-only**: every picture in an R1 set satisfies R1, and similarly for R2 and R3. Each set contains examples of all five A/B/C configurations that satisfy its rule.

Certain images are particularly revealing:

| Configuration | Unique rule it satisfies |
|---|---|
| `001`: only C present | R1 |
| `010`: only B present | R2 |
| `100`: only A present | R3 |

So a model might spot a decisive image and answer correctly **without evaluating all ten pictures**. We are deliberately keeping that original task for a controlled pilot. **Better accuracy would not, by itself, prove that Qwen learned multi-image logical reasoning.**

## 3. Where do the 90 training examples come from?

We already have **1,000 unique image files**. We are not drawing or generating 900 new pictures.

The original dataset contains 125 images for each of the eight possible A/B/C combinations. We keep its original **800 training images** and divide the other 200 into two groups using the fixed random seed `2026`.

| Purpose | Distinct image files | Ten-image sets made from those files | Sets for R1 / R2 / R3 |
|---|---:|---:|---|
| Training | 800 | **90** | 30 / 30 / 30 |
| Validation | 100 | **30** | 10 / 10 / 10 |
| Final test | 100 | **30** | 10 / 10 / 10 |

**An image file is not a training example. A ten-image set is one training example.**

Why can 100 validation images produce 30 ten-image sets, which contain 300 image appearances? Because we allow an image to appear in **different sets within the same partition**. But no image can appear in two partitions. We also do not repeat an image **inside the same set**.

This matters for interpreting results: 30 validation sets are **not 30 completely independent collections of pictures**.

### How the code actually builds a set

Open [`build_training_sets.py`](../../code/04-lora/build_training_sets.py). Its `build()` function:

1. Reads the filenames, A/B/C labels and original split from `data-1000/metadata.json`.
2. Uses seed `2026` to split the original held-out images evenly into validation and test, keeping the eight A/B/C configurations balanced.
3. Chooses one image of **each of the five satisfying configurations** for the intended rule.
4. Adds **five more different satisfying images** to reach ten.
5. Shuffles their order and assigns a candidate-rule mapping to X/Y/Z.
6. Repeats this until it has 30 training sets per rule, and 10 validation and 10 test sets per rule.

It avoids duplicate ten-image collections even if the same ten filenames would have appeared in a different order. Each set also records which image filenames it shares with other sets in its partition.

The answer labels are deliberately balanced in training: for each rule, the correct answer is X in ten training sets, Y in ten and Z in ten. **The current training loop uses the mapping saved in each set; it does not reshuffle a set's options on every epoch.**

### How do we know the sets are valid?

The builder saves a file called the **manifest**. Think of it as the complete record of the experiment: every set ID, which ten filenames are used, the generating rule, the right answer letter, the split, the seed and image fingerprints.

The `load()` function does not blindly trust that file. It **rebuilds the sets from the actual image metadata and seed**, then checks whether they match the saved manifest. It also checks that all expected image files exist.

The construction ensures that each set includes the rule's decisive configuration, so its generating rule is uniquely identifiable among R1/R2/R3. A validation failure stops the run rather than silently changing the labels or image split.

### What is an epoch, then?

**One epoch means one pass through all 90 training sets.** It is not ten images and it is not 90 epochs.

- **One training example:** 10 images + prompt → one correct answer label.
- **One epoch:** all 90 training examples once.
- **Two epochs:** 180 example presentations; many pictures appear more than once.
- **Batch size 1:** the model processes one ten-image example at a time.
- **Gradient accumulation 4:** after up to four examples, the code combines their training signals and updates the adapter weights once.

With 90 sets, that is **23 optimizer updates per epoch** (the last update covers two examples), or **46 updates over two epochs**. The code shuffles set order each epoch.

## 4. What does LoRA actually train?

Qwen3.5-4B already has pretrained weights: a very large collection of numbers learned before our experiment. Updating all those numbers would be expensive.

**LoRA** keeps those original weights frozen and inserts small **trainable adapter matrices** into selected parts of the model. During training, we change only these added matrices.

**QLoRA** additionally stores the pretrained model in a compressed **4-bit** form, which reduces the memory needed for the frozen weights. It does **not** mean the LoRA adapters are only four bits.

Our first experiment attaches LoRA to the supported linear modules **throughout all 32 language decoder layers**. It does **not** train the vision encoder, and it does **not** update every original weight in the model.

The code finds the actual supported linear modules inside each decoder block. Qwen3.5's blocks are not all identical, so we must discover their real names rather than assume they all use the same attention structure.

| Setting | Current value | Plain-English meaning |
|---|---|---|
| Base model | Qwen3.5-4B | The pretrained model we start from |
| QLoRA precision | 4-bit NF4 | Compress frozen base weights to save GPU memory |
| **Rank** | **8** | Size/capacity of the small LoRA updates |
| Alpha | 16 | Scaling applied to those LoRA updates |
| Dropout | 0.05 | Randomly drop a small fraction of adapter input during training |
| Epochs | **2** | Two passes through the 90 training examples |
| Learning rate | 0.0002 | Size of the optimizer's updates |
| Optimizer | AdamW | Chooses how adapter weights are updated from the loss |
| Per-device batch | 1 | One ten-image example processed at a time |
| Gradient accumulation | 4 | Combine training signals from up to four examples per update |
| Validation frequency | Every 10 optimizer updates and at the end | Check whether the model is improving on unseen examples |

**Why rank 8?** It is a reasonable small starting point, not a rank proved optimal for Boolean logic. Rank 4 or 16 would be a different experiment setting. We begin with 8 to keep this pilot manageable.

The actual training code is [`train_lora.py`](../../code/04-lora/train_lora.py). It records every adapter target and checks that no unrelated original model parameter is trainable.

## 5. How does the model learn from a correct answer?

This is the part called **the loss**.

Suppose our ten images follow R2, and R2 is shown as option Y. We want Qwen to become more likely to produce **Y**.

Imagine Qwen initially assigns the correct answer only a small probability. The training loss is large. If it assigns Y a higher probability after training, the loss becomes smaller.

The code uses the **negative log-likelihood of the correct answer tokens**:

~~~text
Training loss = -log P(correct answer | ten images, prompt)
~~~

This is a standard way to train a language model to prefer the correct answer. It does not tell Qwen *how* to reason; it only tells it what answer was correct.

There is one important distinction from our graphs in Experiment 03:

- **During training**, the answer's probability is calculated using the model's **full vocabulary**, not only X/Y/Z. Only the correct answer tokens contribute to the loss; the prompt itself is not a target.
- **During evaluation**, the code compares the three permitted answers X, Y and Z. Their scores are converted into three probabilities that sum to one. It then chooses the most likely rule.

The code also handles an answer label that might require more than one token. It checks the right token positions so it does not accidentally train on the prompt or on the wrong answer token.

The steps for each training set are:

~~~text
Read ten images and their correct generating rule
                  |
                  v
Make the prompt and find the correct X/Y/Z label
                  |
                  v
Qwen processes the actual images and the text
                  |
                  v
Calculate loss on the correct answer token(s)
                  |
                  v
Backpropagate the error into LoRA adapters only
                  |
                  v
Update adapters; repeat for the next examples
~~~

This is supervised fine-tuning. There is **no separate reward model** and no training on free-text explanations.

## 6. Why do we need a validation set and a test set?

We have to avoid grading the model only on the examples it just learned.

- **Training sets (90):** used to update LoRA weights.
- **Validation sets (30):** used to decide which saved adapter is best. They are not used to update weights.
- **Test sets (30):** set aside until the training setup is fixed. They give our final before/after comparison.

For validation, the program checks **all three cyclic answer-option rotations** for each set:

| Rotation | X | Y | Z |
|---|---|---|---|
| 0 | R1 | R2 | R3 |
| 1 | R2 | R3 | R1 |
| 2 | R3 | R1 | R2 |

Thus, **30 validation sets × 3 rotations = 90 prediction evaluations per validation check**. The same 30 sets are being seen with different answer mappings; these are not 90 independent examples.

After every ten optimizer updates, and at the end, training measures the validation loss and keeps the adapter with the **lowest validation loss**. This saved adapter is called `best_adapter/`. The last training step is not automatically the best checkpoint.

We should not inspect final-test accuracy repeatedly to decide how many epochs to train or which settings to use.

## 7. What do we compare before and after LoRA?

First, we run the **original pretrained Qwen** on the validation sets and save its answers. This is the **baseline**.

Then we train QLoRA, load the best adapter, and ask the **same validation questions** again.

Only after choosing settings do we use the untouched test sets. Both models see exactly the same ten-image sets and three answer-option rotations.

Open these scripts:

| File | What it does |
|---|---|
| [`evaluate_lora.py`](../../code/04-lora/evaluate_lora.py) | Runs rule prediction on either the pretrained model or the fine-tuned model |
| [`compare_predictions.py`](../../code/04-lora/compare_predictions.py) | Reads the two saved result files and calculates the before/after differences; no model needs to be loaded |
| [`common.py`](../../code/04-lora/common.py) | Reuses the original model wrapper and prompt, prepares the images, scores X/Y/Z and checks that run settings match |

We record:

- **Overall accuracy:** fraction of predictions matching the generating rule.
- **Per-rule accuracy:** did it improve for R1, R2 and R3 equally?
- **Answer-label bias:** does the model prefer X, Y or Z regardless of the pictures?
- **Rule bias:** does it overpredict R1, R2 or R3?
- **Confusion matrix:** which rules does it confuse with which others?
- **Changes on the same examples:** which wrong answers became correct, and which correct answers became wrong?
- **Probability change:** did the correct rule's score increase, even when the chosen answer stayed the same?

For example, if the pretrained model predicts R1 for a true R2 set, but the LoRA model predicts R2, that is one corrected prediction. If the reverse happens on another set, that is one new error. **We need both counts, not merely a final accuracy percentage.**

Random guessing among three balanced rules gives **1/3 accuracy**. But merely beating chance does not establish that Qwen used all ten images or computed the full rule.

The separate [Experiment 05 guide](https://github.com/SathyaHamsiniP/looking-beyond-attention/blob/feature/05-post-lora-patching/three-operand-experiments/experiments/05-post-lora-comparison/README.md) explains the **second comparison**, which tests activation patching before and after training.

## 8. What are the scripts and saved files?

~~~text
code/
  04-lora/
    build_training_sets.py   Make and check the image sets
    common.py                Shared prompt, model and scoring helpers
    train_lora.py            Train the LoRA adapters
    evaluate_lora.py         Get original or LoRA rule predictions
    compare_predictions.py   Compare those predictions without a GPU
    requirements.txt         Needed Python packages
~~~

The Experiment 05 patching scripts live on the separate `feature/05-post-lora-patching` branch; they are deliberately absent here until Experiment 04 results have been committed. Everything is saved in **new run folders**. We never overwrite the completed Experiment 03 results.

| Saved file or folder | What to look for |
|---|---|
| `rule_identification_manifest.json` | Exact image splits, set membership and correct answer labels |
| `config.json`, `manifest.json` | Exact settings and data used for one run |
| `trainable.json` | Which adapter parameters were trained, and how many |
| `losses.jsonl` | Training loss and occasional validation loss at each optimizer update |
| `selection.json` | Which checkpoint had the lowest validation loss |
| `best_adapter/` | Saved adapter to use for evaluation |
| `checkpoint-N/` | Adapter plus optimizer and random state for resuming training |
| `predictions.jsonl` | One line per set and answer rotation: true rule, predicted rule, probability scores |
| `metrics.json` | Accuracy, per-rule accuracy, option bias and confusion counts |
| `matched_predictions.csv` | Which answers changed from pretrained to LoRA |
| `comparison.json` and PNG plots | Before/after summary, rule accuracy and biases |

A `.jsonl` file has **one JSON record per line**. A `.csv` is a table you can open in a spreadsheet. A `.png` is a graph. The model is **not needed** to read saved metrics or generate comparison graphs.

The program prints a run folder after starting. It adds a short identifier based on the settings, dataset and software versions. Copy the **printed folder path** rather than guessing it. Changing code or package versions may change that identifier.

## 9. How to run Experiment 04 in Colab

These are the actual script interfaces on this branch. Steps **9.1 and 9.2 are CPU work**; loading Qwen, training and evaluation need a suitable GPU. The comparison script only needs saved files and a CPU.

### 9.1 Get the code, connect Drive and prepare paths

Select a GPU runtime in Colab (for example T4). In a Python cell:

~~~python
from google.colab import drive
drive.mount('/content/drive')
~~~

In a new Colab cell, put `%%bash` on the first line, then:

~~~bash
git clone --branch feature/04-simple-lora-patching https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
cd /content/looking-beyond-attention
pip install -r three-operand-experiments/code/04-lora/requirements.txt
~~~

If Colab asks you to restart after package installation, restart, remount Drive and continue with the existing checkout.

Use these paths in a shell cell:

~~~bash
cd /content/looking-beyond-attention
export CODE=three-operand-experiments/code
export DATA='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000'
export RUNS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results/04-simple-lora'
export PATCH_RUNS='/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results/05-simple-lora-patching'
mkdir -p "$RUNS" "$PATCH_RUNS"
export MANIFEST="$RUNS/rule_identification_manifest.json"
~~~

**Important for Colab:** Each `%%bash` cell starts a new shell. Re-run the path and `export` block at the top of **each new shell cell**, or execute related commands in one cell. The command snippets below assume these variables have been set in that shell.

### 9.2 Build and check the examples (no GPU needed)

~~~bash
python "$CODE/00-data-generation/download_dataset.py" --output-dir "$DATA"
python "$CODE/04-lora/build_training_sets.py" --data-dir "$DATA" --manifest "$MANIFEST"
python "$CODE/04-lora/build_training_sets.py" --data-dir "$DATA" --manifest "$MANIFEST" --validate-only
~~~

If the dataset already exists, the downloader checks it without replacing it. The builder checks the real files and should report **800/100/100 images** and **90/30/30 ten-image sets**.

### 9.3 Fix the exact Qwen version

Online model weights can be updated. For a meaningful comparison, the pretrained model used before training must be the **same original version** used after training. We save a fixed 40-character model commit ID.

~~~bash
if [ ! -f "$RUNS/model_revision.txt" ]; then
  python -c 'from huggingface_hub import model_info; print(model_info("Qwen/Qwen3.5-4B").sha)' > "$RUNS/model_revision.txt"
fi
export REV="$(cat "$RUNS/model_revision.txt")"
~~~

Keep `REV` the same for **every** original-model, LoRA-model and activation-patching command. Do not replace it with `main`. Keep the image resolution (`--max-pixels`) and precision the same too.

### 9.4 First check whether training can run on the GPU

~~~bash
python "$CODE/04-lora/train_lora.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$RUNS/training" --smoke-test
~~~

A **smoke test** is a tiny real training run that catches loading, tokenization and memory problems before we commit to the complete experiment. Here it uses just two training sets, up to two optimizer updates, and two validation sets. **It still processes all ten images in each set.**

A smoke test finishing does **not** prove that the full run will fit or finish. Never use smoke-test results as research measurements.

If memory runs out, reduce `--max-pixels` or `--rank` explicitly. Do not silently drop images. If resolution changes, restart the pretrained and LoRA comparisons with the **same new resolution**. The `--max-tokens` limit rejects excessively long inputs; it does not silently shorten them.

### 9.5 Measure the original model before training

~~~bash
python "$CODE/04-lora/evaluate_lora.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$RUNS/pretrained" \
  --partition validation
~~~

This evaluates **30 validation sets × 3 answer orders = 90 predictions**. It prints the exact results folder. Copy it:

~~~bash
export BASELINE='/paste/the/printed/pretrained/validation/run/folder'
~~~

Do **not** substitute a smoke-test folder.

### 9.6 Fine-tune the LoRA adapters

~~~bash
python "$CODE/04-lora/train_lora.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$RUNS/training" \
  --baseline-run "$BASELINE" --epochs 2
~~~

This is the **full** two-epoch run with rank 8 and the other settings from Section 4. It refuses to start unless the supplied original-model baseline is complete and matches the manifest and model settings.

Training saves its losses, checkpoints and `best_adapter/`. Set:

~~~bash
export ADAPTER='/paste/the/printed/training/run/folder/best_adapter'
~~~

To resume interrupted training, rerun with identical settings and `--resume /full/run/checkpoint-N`. Use a trusted checkpoint from this experiment, preferably the latest one. The saved checkpoint contains more than adapter weights: it also stores the optimizer, step number and random state.

### 9.7 Evaluate LoRA on validation, then use the final test only once settings are fixed

~~~bash
python "$CODE/04-lora/evaluate_lora.py" \
  --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$RUNS/finetuned" \
  --adapter "$ADAPTER" --partition validation
~~~

Once you are satisfied with the setup, evaluate the **same final-test examples** on both models:

~~~bash
python "$CODE/04-lora/evaluate_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$RUNS/pretrained" --partition test

python "$CODE/04-lora/evaluate_lora.py" --manifest "$MANIFEST" --data-dir "$DATA" \
  --revision "$REV" --output-root "$RUNS/finetuned" \
  --adapter "$ADAPTER" --partition test
~~~

### 9.8 Compare the two saved prediction runs (no GPU required)

Use the **exact run folders** printed by the two test evaluations:

~~~bash
python "$CODE/04-lora/compare_predictions.py" \
  --baseline /full/pretrained/test/run \
  --finetuned /full/finetuned/test/run \
  --output "$RUNS/prediction-comparison-test"
~~~

The script rejects incomplete runs or different inputs/settings. Its output folder must be **new**. For a preliminary comparison, use matching **validation** folders instead.

After completing Experiment 04 and pushing its results, switch to the separate branch and follow [Experiment 05](https://github.com/SathyaHamsiniP/looking-beyond-attention/blob/feature/05-post-lora-patching/three-operand-experiments/experiments/05-post-lora-comparison/README.md) to repeat activation patching with both models and compare the layer-wise changes.

## 10. What should we expect, and what would the findings mean?

We have **not** established that LoRA improves accuracy. Several outcomes are possible:

- **Validation and test accuracy improve:** the adapter helps with rule identification on these examples. It may still use decisive-image shortcuts.
- **Training loss decreases but held-out accuracy does not:** the model may be fitting training examples without generalizing.
- **Accuracy improves but predictions still favour one letter/rule:** LoRA has not removed all answer bias.
- **Accuracy hardly changes but answer probabilities do:** the model's confidence changed without changing the winning answer.
- **Results differ across option rotations:** the model may be reacting to answer labels or presentation order.

Even the best outcome would be an observation about **this three-rule, positive-only, image-reusing task**. It would not, on its own, establish general Boolean reasoning.

For our research question, the next step is to ask whether **activation patching changes in a more rule-specific way after LoRA**. That is what [Experiment 05](https://github.com/SathyaHamsiniP/looking-beyond-attention/blob/feature/05-post-lora-patching/three-operand-experiments/experiments/05-post-lora-comparison/README.md) is designed to test.

## 11. Quick checks and important limits

From the repository checkout:

~~~bash
python -m unittest discover -s three-operand-experiments/tests -v
~~~

CPU tests cover data generation, label mapping, answer-token alignment and several model/adapter behaviours with small test models. They are **not** a substitute for running the actual 4B model and its ten images on a real GPU.

- **No completed GPU training results are published yet.**
- A Colab T4 has limited GPU memory; even four-bit QLoRA with ten images might not fit.
- Training-set image reuse and repeated answer rotations reduce how independent the measured examples are.
- Changing the image resolution, model revision, prompt or data split means the old comparison is no longer directly matched.
- Do not put model checkpoints, image datasets or new GPU run folders into Git.
- Do not overwrite or reinterpret the existing Experiment 03 baseline results as this experiment's matched pretrained results.
- This implementation intentionally does **not** train only one selected decoder layer. That would be a **separate experiment requiring a justified layer-selection rule**.

**Where to read next:** [Experiment 05 — What changes in activation patching after LoRA?](https://github.com/SathyaHamsiniP/looking-beyond-attention/blob/feature/05-post-lora-patching/three-operand-experiments/experiments/05-post-lora-comparison/README.md).

## Check that AMP really updates the adapter

FP16 training uses a gradient scaler. It multiplies the loss before backward, then divides the gradients before AdamW. A very large scale can overflow FP16 gradients. When that happens, the scaler skips the optimizer update and lowers its scale. Finishing two forward/backward passes therefore does not mean training worked.

The starting scale is now **16**, rather than PyTorch's default 65,536. Set `--initial-gradient-scale` to change it. We inspect gradients after unscaling and before clipping. Infinite or NaN gradients are not clipped or applied to the weights.

Each `losses.jsonl` record includes:

- `step`: attempted update number.
- `optimizer_step_occurred`: whether AdamW actually ran, checked with an optimizer hook.
- `optimizer_step_number`: count of real updates so far.
- `gradient_scale_before` and `gradient_scale_after`.
- `nonfinite_gradient_values` and `nonfinite_gradient_parameters`.
- `missing_gradient_parameters` and the gradient norm before clipping.
- `lora_b_matrices_changed`: how many B matrices changed in that attempt.
- `optimizer_state_entries`: how many parameters have optimizer state.

A smoke test fails if no real updates occur or no B matrix changes. It does not write a successful completion marker in that case. The log remains available to explain the failure. A successful test need not change every B matrix in two steps. Check both the update flags and the weight-change counts. `complete.json` records total real updates and steps with B changes.

### Retry without touching the failed run

Use the failed run's `config.json` to keep its exact manifest, ten-image preprocessing, pinned model revision and training settings. Do not resume the old failed checkpoint for this check. The new scale is part of the saved settings and gives a separate run ID.

In a GPU Colab runtime, after mounting Drive, run this setup cell. Replace the two paths with your actual saved folders:

```python
%cd /content/looking-beyond-attention
!git switch feature/04-simple-lora-patching
!git pull --ff-only origin feature/04-simple-lora-patching

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FAILED_RUN = Path('/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results/04-simple-lora/training/PASTE_FAILED_RUN_ID')
BASELINE_RUN = Path('/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results/04-simple-lora/pretrained/PASTE_VALIDATION_RUN_ID')
old = json.loads((FAILED_RUN / 'config.json').read_text())
assert old['precision'] == '4bit', 'Use the same four-bit setup as the failed run.'
FRESH_ROOT = FAILED_RUN.parent / ('amp-check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
FRESH_ROOT.mkdir(exist_ok=False)
script = 'three-operand-experiments/code/04-lora/train_lora.py'
keep = ['manifest', 'data_dir', 'revision', 'precision', 'max_pixels', 'rank', 'alpha',
        'dropout', 'learning_rate', 'gradient_accumulation', 'validation_frequency',
        'checkpoint_interval', 'max_tokens']
command = [sys.executable, script]
for key in keep:
    command += ['--' + key.replace('_', '-'), str(old[key])]
command += ['--initial-gradient-scale', '16', '--output-root', str(FRESH_ROOT / 'smoke')]
subprocess.run(command + ['--smoke-test'], check=True)
print('Fresh verification folder:', FRESH_ROOT)
```

This starts only the two-step smoke test. A failure stops the cell. If both attempts still overflow, inspect the parameter names in the log. Try `--initial-gradient-scale 1` in another fresh folder. A lower scale cannot fix gradients that are already non-finite before scaling; do not treat repeated skips as success.

Inspect the check in a separate cell:

```python
smoke_file = next((FRESH_ROOT / 'smoke').glob('*/complete.json'))
smoke_result = json.loads(smoke_file.read_text())
print(smoke_result)
assert smoke_result['optimizer_updates'] > 0
assert smoke_result['steps_with_lora_b_changes'] > 0
for line in (smoke_file.parent / 'losses.jsonl').read_text().splitlines():
    print(json.loads(line))
```

Only after that passes, run this **separate cell** to start the full two-epoch training. It uses the same manifest, revision, resolution and rank, and requires the existing pretrained validation baseline:

```python
assert BASELINE_RUN.is_dir(), 'Set the complete pretrained validation folder first.'
full_command = command[:]
index = full_command.index('--output-root')
full_command[index + 1] = str(FRESH_ROOT / 'full-two-epochs')
subprocess.run(full_command + ['--epochs', '2', '--baseline-run', str(BASELINE_RUN)], check=True)
```

No full training is started by the smoke-test or inspection cells. The older results remain untouched. The lower starting scale and the skip checks have CPU tests, but the actual four-bit T4 smoke test must still be run before claiming the overflow is resolved on Qwen.
