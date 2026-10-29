# Three-operand Boolean rule identification in Qwen VLMs

Extension of the operator-controls experiment in Roshan's *looking-beyond-attention* repository (README experiment 04; script `exp8_operator_controls.py`). This repository compares **Qwen3-VL-4B-Instruct** and **Qwen3.5-4B** on three-operand Boolean rule identification.

## Research question

Given ten positive CLEVR-style scenes sampled from a hidden Boolean rule, can the model identify the generating expression among three possibilities?

- **R1:** `(A AND B) OR C`
- **R2:** `(A AND C) OR B`
- **R3:** `(B AND C) OR A`

A denotes a green cube, B a blue sphere, and C a red cylinder. See the experiment-specific README for sampling, rule identifiability, controls and caveats.

## Findings

| Test | Qwen3-VL-4B | Qwen3.5-4B |
|---|---:|---:|
| Main 72 image trials | 33.3% | 31.9% |
| Explicit Boolean gate evaluation (24 cases) | 79.2% | 95.8% |
| Perception check (8 images) | 8/8 | 8/8 |

Chance accuracy for the main three-way rule choice is 33.3%. Both models showed pronounced answer-position bias. Better direct Boolean gate evaluation did not translate into better multi-scene rule identification on this benchmark.

## Structure

```text
notebooks/exp04/Qwen3_VL_4B.ipynb
notebooks/exp04/Qwen3_5_4B.ipynb
experiments/04-three-operand-operator-controls/README.md
data/metadata.json
data/images/                  # 48 original scenes
results/qwen3-vl-4b/<condition>/{config.json,runs.jsonl,summary.json}
results/qwen3.5-4b/<condition>/{config.json,runs.jsonl,summary.json}
```

The Colab notebooks are the main implementations; there is no separate `code/` tree. Their concise progress messages show the true and predicted rule, correctness, and choice position. Internal trial IDs remain unchanged in the JSONL outputs. Saved notebook cell outputs have been cleared for GitHub readability.

## Running in Colab

Open one of the notebooks in `notebooks/exp04/`, select a GPU runtime and execute setup and definitions before enabling any desired experiment. The notebooks currently expect the assembled Drive project at:

`MyDrive/LogicCBM++/three-operand-experiments/`

They read images from `data/` and save their runs under `results/<model>/`. The Qwen3-VL notebook additionally loads Roshan's original `looking-beyond-attention*.zip` archive from the **parent** `MyDrive/LogicCBM++/` directory. The original repository code is not redistributed here. The two notebooks have different Transformers requirements; run them in separate Colab sessions.

If an experiment already has matching `config.json` and `runs.jsonl`, its runner resumes rather than rerunning completed trials. To display previously completed runs in the new progress format, read `runs.jsonl`; the notebook's progress messages appear only for newly executed inferences. The result files do not need rewriting.

## Interpretation and provenance

The next-token logits are normalized over just the three allowed answer labels X/Y/Z, not over every possible answer. Many trials reuse the same scene set under different label permutations and should not be considered independent observations. The Qwen3.5 open and guided discovery responses were truncated before final answers could be parsed. These inconclusive runs are not included in this cleaned repository; the original uploaded results ZIP remains archived separately in the parent `LogicCBM++/` folder. **`None` must not be interpreted as 0% reasoning accuracy.** The Qwen3-VL notebook retains its own completed discovery controls.

Both notebooks preserve the experiment prompts, deterministic sampling, model-specific inference and analysis logic. Progress messages and Drive paths have been updated; the original JSONL results are unchanged. The code was structurally checked but not rerun on a GPU as part of the repository cleanup.

## Attribution

Based on Roshan and collaborators' operator-controls study in *looking-beyond-attention*. Please consult the upstream code and license/permissions before redistributing the underlying model-wrapper or readout source.
