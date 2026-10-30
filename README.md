# Looking Beyond Attention

A vision-language model might recognize everything in an image and still struggle to reason about what it sees. This project explores that gap.

We're interested in understanding whether vision-language models actually use the information available to them when solving logical problems, or whether their predictions are influenced by simpler patterns and biases.

Rather than relying only on attention weights to understand model behavior, we use controlled experiments to test what information a model recognizes, how it combines evidence, and how its answers change when that evidence is modified.

## What we're working on

Our first set of experiments focuses on **three-operand Boolean rule identification**.

We generate CLEVR-style images containing combinations of three concepts:

- **A:** Green cube
- **B:** Blue sphere
- **C:** Red cylinder

Each set of images is generated using one of three Boolean rules. The model sees ten examples and must identify which rule explains them.

We tested this with **Qwen3-VL-4B-Instruct** and **Qwen3.5-4B**.

To understand what might be going wrong, we also tested whether the models could recognize the objects, evaluate Boolean operations, use decisive examples, and remain consistent when answer choices were rearranged.

[Read more about the experiments](three-operand-experiments/experiments/04-three-operand-operator-controls/README.md)

## What we found

| Task | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Rule identification | 33.3% | 31.9% |
| Object recognition (8 images) | 100% | 100% |
| Boolean gate evaluation (both stages correct) | 79.2% | 95.8% |

Both models recognized the objects correctly in our small perception test and performed considerably better when asked to evaluate explicit Boolean operations.

However, neither reliably identified the rule behind a set of images. Their accuracy remained close to the three-choice chance level of 33.3%.

We also observed strong answer-position biases. For example, Qwen3-VL selected the first displayed option in 143 out of 144 trials in our option-order experiment.

These findings suggest that **recognizing concepts and evaluating individual logical expressions do not necessarily translate into inferring a rule from multiple examples**.

The experiments are small and controlled, so we don't treat these observations as a general verdict on the models' reasoning abilities.

## Repository structure

- `three-operand-experiments/code/` — Python implementations for both models and their dependencies
- `three-operand-experiments/data/` — Generated images and metadata
- `three-operand-experiments/tests/` — Automated checks for the experimental code
- `three-operand-experiments/results/` — Recorded predictions and experiment summaries
- `three-operand-experiments/experiments/` — Experimental methodology and findings

The experiments are implemented as Python scripts and can be run on a machine with suitable GPU resources. Each model has its own requirements file.

The Qwen3.5 Python runner has been tested with actual model inference. The converted Qwen3-VL runner has passed CPU-based tests but still needs a separate GPU verification.

## Project status

This is ongoing research. We're using these initial experiments to understand the difference between perceiving evidence, carrying out logical operations, and identifying relationships across multiple examples.
