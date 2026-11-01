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
representations. Both Colab runs are complete according to the experiment
records reported by the researcher; their output files are still on Drive and
have not yet been checked into this repository. We keep those reported findings
separate from the verified operator-control results.

The next question is whether the correct **rule**, rather than just the objects,
is readable internally after ten images. Rule probing is next. Activation
patching will follow as experiment 03, to test whether changing representations
changes the final prediction. Neither is implemented yet.

## Experiments

[01 — Operator controls](three-operand-experiments/experiments/01-operator-controls/README.md)
explains the task, controls and recorded results.

[02 — Probing](three-operand-experiments/experiments/02-probing/README.md)
explains the concept probes, their limits, Drive result transfer and exact Colab
commands for both models.

Code, experiment notes and curated results use the same experiment numbers under
`three-operand-experiments/`. The original 48 images and metadata remain in
`data/`. Model requirements are shared across experiments in `code/`. GPU runs
use Colab; activation caches stay on Drive. Existing operator-control JSONL
records are preserved, while new curated probing outputs are CSV, JSON and PNG.

## Acknowledgements

This project extends Roshan and collaborators' two-operand experiments. The
Qwen3-VL loader still uses their external model wrapper, and the probing hooks
follow their layer-pooling approach. The reference implementation must be
provided separately for Qwen3-VL runs. Qwen3.5 uses the maintained native loader.
These different preprocessing and inference paths matter when comparing models.
