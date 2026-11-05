# Compare activation patching before and after LoRA

This is the activation-patching part of the Experiment 04 study. It asks whether changing one internal vector affects the model's answer differently after training. Use the [training guide](../04-lora/README.md) to build the fixed sets and obtain the adapter first.

A receiver is the input we change. A donor supplies a vector from the same layer and token position. We record both original answers, patch the receiver, and score it again. The model weights stay fixed during this test. See the [Experiment 03 guide](../03-activation-patching/README.md) for the basic idea.

## Why a new baseline?

The completed Experiment 03 used a different collection of held-out sets. Its 3,456 trials are a methodological reference, not a matched comparison for the new training split. We therefore run the pretrained model again on exactly the sets and pairs used for the adapter model. We do not change or reuse its results folder.

Use validation inputs while choosing settings. Use final-test inputs only once the settings are fixed. Compare both models on the same partition.

## Choose a small pair plan

`--receivers-per-rule 1` selects one receiver for each rule. Every receiver gets one donor from each rule, including its own. That is nine pairs: three same-rule and six different-rule. The code chooses the donor with the fewest shared image files, preferring zero overlap when possible. Shared filenames are recorded rather than hidden. Identical receiver and donor sets are never paired.

The same pair plan is saved for both models. There are three answer rotations and both final/previous-token positions. The default layers are 1, 8, 16, 24 and 32. This small run has 9 × 3 × 5 × 2 = 270 patching trials per model. It still needs additional normal passes and can take time. `--layers` accepts any selected 1-based layer numbers. A full sweep can pass all 32 numbers explicitly.

## Commands

Use the `CODE`, `DATA`, `MANIFEST`, `REV`, `ADAPTER` and `PATCH_RUNS` variables from the training guide. These are GPU commands. First run matched smoke tests, which use one same-rule pair, one different-rule pair, and layers 1 and 32. Ten images are still supplied to each input.

```bash
python "$CODE/05-post-lora-comparison/patch_finetuned.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$PATCH_RUNS/pretrained" --partition validation --smoke-test
python "$CODE/05-post-lora-comparison/patch_finetuned.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$PATCH_RUNS/finetuned" --adapter "$ADAPTER" --partition validation --smoke-test
```

For the selected-layer run, omit `--smoke-test`:

```bash
python "$CODE/05-post-lora-comparison/patch_finetuned.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$PATCH_RUNS/pretrained" --partition validation --receivers-per-rule 1 --layers 1 8 16 24 32
python "$CODE/05-post-lora-comparison/patch_finetuned.py" --manifest "$MANIFEST" --data-dir "$DATA" --revision "$REV" --output-root "$PATCH_RUNS/finetuned" --adapter "$ADAPTER" --partition validation --receivers-per-rule 1 --layers 1 8 16 24 32
```

After choosing settings, repeat both commands with `--partition test`. Do not select a layer based on test results. Both models must use identical layer, position, resolution, precision and revision flags. Use `--validate-only` to check the plan on CPU first.

Use the two printed run paths in this CPU command:

```bash
python "$CODE/05-post-lora-comparison/compare_patching.py" --baseline /full/pretrained/patching/run --finetuned /full/finetuned/patching/run --output "$PATCH_RUNS/comparison-validation"
```

Choose a new output folder for each comparison. The script refuses mismatched or incomplete runs. It checks the manifest, exact pair plan, layers, positions, mappings, model revision and preprocessing. It includes all selected receivers, not only ones where an adapter succeeds.

## What is saved?

Each model run saves its configuration, complete manifest, pair plan and `patches.jsonl`. Every trial records full original receiver/donor probability vectors, patched probabilities, predictions, original correctness and answer changes. The configuration records the adapter identity as file checksums. Trials are keyed by pair, rotation, layer and position for safe resumption.

Comparison writes `before.json`, `after.json`, `differences.csv`, source references and six plots:

- Signed donor true-rule probability change: did its score rise or fall?
- Absolute donor-rule change: how large was the change, ignoring direction?
- Receiver true-rule change: did support for the receiver's rule rise or fall?
- Answer-change rate: did the chosen rule change?
- Donor true-rule transfer: did the answer change to the generating rule?
- Donor predicted-answer transfer: did it change to what the donor chose, even if wrong?

Rates use all trials in the condition/layer/position group as their denominator. Same-rule and different-rule targets mean different things, so raw full vectors and receiver-rule changes remain available. Signed effects can cancel even when individual absolute changes are large.

## What can we conclude?

No new GPU results are available yet. The historical baseline's layer-6 signed mean was least negative; layer 32 strongly copied donor predictions. Neither observation established a reasoning layer. The new experiment must also keep correct rule transfer separate from copying a biased or wrong donor answer.

Patching shows that a change can affect the answer. It does not by itself identify a Boolean calculation. Shared images within a partition and repeated option rotations limit independent evidence. Report accuracy, bias and controls together with patching effects. Do not call the layer with the biggest effect the reasoning layer.
