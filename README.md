# Looking Beyond Attention

Can a vision-language model work out a Boolean rule from examples? We wanted to
understand whether recognizing what is in an image is enough to reason about a
set of images.

We extended an earlier two-operand task to three concepts: a green cube, a blue
sphere and a red cylinder. We showed Qwen3-VL-4B and Qwen3.5-4B ten images and
asked which of three Boolean rules explained them. Both models recognized the
objects well and did much better when we gave them explicit Boolean operations.
But their rule-identification accuracy stayed close to chance. Changing answer
positions had a large effect, while removing decisive evidence changed output
probabilities very little in our controls.

That led us to look inside the models. Our concept probes ask whether a small
classifier can recover the eight combinations of these concepts from internal
representations. Both Colab runs are complete. Their outputs stay on Google Drive; the
[probing notes](three-operand-experiments/experiments/02-probing/README.md) summarize
the measured scores reported from those runs and their limitations.

The next question is whether the correct **rule**, rather than just the objects,
is readable internally after ten images. The Qwen3.5 rule-probing pilot reached 100% on three held-out sets, but simple
scene-based baselines did too. We are now testing paired positive/negative
evidence to ask whether predictions respond to information from multiple images. Activation patching will follow as experiment 03 to
test whether changing representations changes the final prediction.

## Experiments

[01 — Operator controls](three-operand-experiments/experiments/01-operator-controls/README.md)
explains the task, controls and recorded results.

[02 — Probing](three-operand-experiments/experiments/02-probing/README.md)
explains the concept probes, their limits and exact Colab
commands for both models.

Code and experiment notes use the same experiment numbers under
`three-operand-experiments/`. The original 48 images and metadata remain in
`data/`. Model requirements are shared across experiments in `code/`.

## Where outputs go

GitHub holds code and documentation. All new experiment outputs stay on Google
Drive, including CSVs, configurations, plots and activation checkpoints. Existing
tracked results are preserved as an archive; there is no copying or syncing step.

Scripts default to the existing results root:
`/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results`.
Mount Drive before running. If it is mounted differently, pass the actual
existing root using `--experiment-dir`. Scripts check it before saving and never
create a missing results root. Dataset-only `--validate-only` checks do not need
Drive. Experiment subfolders are created only for the run being executed.

Future scripts should use the same default and root check in `code/results_paths.py`,
then save under the relevant experiment subfolder. Rule probing uses
`probing/<model>/rule/`; future patching will use `activation_patching/<model>/`. Do not create their folders in advance. Findings
belong in the experiment READMEs, with the measured scores and limitations;
output files do not need to be uploaded to GitHub.

## Acknowledgements

This project extends Roshan and collaborators' two-operand experiments. The
Qwen3-VL loader still uses their external model wrapper, and the probing hooks
follow their layer-pooling approach. The reference implementation must be
provided separately for Qwen3-VL runs. Qwen3.5 uses the maintained native loader.
These different preprocessing and inference paths matter when comparing models.
