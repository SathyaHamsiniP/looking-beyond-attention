# 04 — Bottleneck controls

**Purpose:** distinguish scene perception, image-label binding, evidence combination, candidate verification, and answer-format bias. Reuse the three held-out evidence-integration families from Experiment 02.

## Quick Colab workflow

Mount Drive, clone/checkout the branch, install the existing Qwen3.5 dependencies:

~~~bash
pip install -r three-operand-experiments/code/requirements-qwen35.txt

python three-operand-experiments/code/04-bottleneck-controls/run_bottleneck_controls.py \
 --data-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data" \
 --validate-only

python three-operand-experiments/code/04-bottleneck-controls/run_bottleneck_controls.py \
 --data-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data" \
 --experiment-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results" \
 --smoke-test

python three-operand-experiments/code/04-bottleneck-controls/run_bottleneck_controls.py \
 --data-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data" \
 --experiment-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results"
~~~

Adjust data/results paths if your Drive structure differs. The smoke test takes **15 model calls**; the full pilot is **87 calls** across only **three underlying families**. The console prints every prediction and a final score summary. Run state automatically resumes from Drive; results are saved under bottleneck-controls/smoke-v2 or bottleneck-controls/quick-v2, with JSONL checkpoints plus CSV and JSON summaries.

To test code without GPU:

~~~bash
python -m unittest discover -s three-operand-experiments/tests -p 'test_bottleneck_controls.py' -v
~~~

## Tests

| Control | What we compare |
|---|---|
| Oracle symbolic rule selection | Can Qwen choose a rule when the four labeled scenes are supplied as exact ABC bits? |
| Original image rule selection | Can it select using four images followed by their labels? |
| Interleaved image rule selection | Does moving the label next to each image improve performance? |
| Candidate verification | Can it say YES or NO to each rule separately (oracle vs. interleaved)? |
| Image-label binding | Can it report the ABC bits and given label for each image? |
| Answer readout | Does asking for R1/R2/R3 versus the complete expression change success? |
| Counterfactual sensitivity | Does the output distribution shift when one decisive diagnostic scene is changed? |

Each held-out family yields 3 answer-layouts × 3 input modes = 9 choices, 12 candidate checks (both YES/NO label orders), 4 individual-image binding tests, and 4 generated readouts, totaling 29 calls per family.

The script computes per-trial correctness against the uniquely valid rule in the both-clue condition, mean correct-rule probability, X/Y/Z distribution, candidate-check accuracy, scene/label accuracy, and paired counterfactual distribution changes.

## Interpreting the results

* Oracle succeeds, images fail: possible visual recognition/binding bottleneck.
* Candidate checks succeed, rule choice fails: investigate answer selection or integration across judgments.
* Oracle fails too: perception cannot be the only issue.
* Answer-format results differ: scoring/prompt format is a serious confound.
* Small counterfactual change: model output does not noticeably track the decisive image change, but output bias can mask this.

**Caution:** only three families with overlapping image configurations. The 87 calls are not 87 independent problems. This is a diagnostic pilot, not a definitive measure of reasoning. Original experiments remain unchanged.

## Important v2 correction (November 2)

The initial smoke run used short free-form generation for YES/NO and four-line binding responses. Qwen produced reasoning text instead of the strict formats; some 0% scores were parsing/truncation artifacts, not demonstrated perception errors. **Do not interpret the old smoke results as valid binding or verification accuracy.** Version 2 uses normalized next-token scores among explicit labels instead of parsing generated explanations. It checks each image independently for its ABC bits and provided label, and swaps the X/Z YES/NO mapping in candidate verification. Version 2 saves into new smoke-v2/quick-v2 folders, keeping the original results intact. Please run validation and smoke again before the full pilot.
