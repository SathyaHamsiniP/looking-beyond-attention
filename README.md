# Looking Beyond Attention

**Controlled experiments on logical reasoning and evidence use in vision-language models.**

This research asks whether a model uses the relevant visual or textual evidence to infer a Boolean rule, rather than responding to superficial cues such as option placement. Attention weights alone are not causal evidence of information flow, so we test perception, Boolean execution, rule identification, option-order robustness, and sensitivity to decisive evidence separately.

## Current study: three-operand operator controls

Ten CLEVR-style scenes are sampled from one of three hidden Boolean rules:

- **R1:** `(A AND B) OR C`
- **R2:** `(A AND C) OR B`
- **R3:** `(B AND C) OR A`

A = green cube; B = blue sphere; C = red cylinder. The main task is **3 rules × 8 sets per rule × 3 cyclic label assignments = 72 scored presentations**, from **24 distinct ten-image sets**.

| Evaluation | Qwen3-VL-4B-Instruct | Qwen3.5-4B |
| --- | ---: | ---: |
| Hidden-rule identification (72 presentations) | 33.3% | 31.9% |
| Both Boolean gate outputs correct (24 cases) | 79.2% | 95.8% |
| Concept recognition (8 held-out images) | 8/8 | 8/8 |

The three-choice chance baseline is **33.3%**. Good concept recognition and single-case Boolean execution did not result in reliable hidden-rule identification. Presentation bias is substantial: in the Qwen3-VL position-order control, the first displayed option was chosen in **143 of 144** trials. This is a small, controlled study; the findings should not be generalized to all reasoning tasks or interpreted as a causal mechanism.

[Detailed experimental protocol, controls and results](three-operand-experiments/experiments/04-three-operand-operator-controls/README.md)

## Repository layout

```text
looking-beyond-attention/
├── .gitignore
├── README.md                        # project overview, setup, and usage
└── three-operand-experiments/
    ├── code/
    │   ├── exp04_qwen3vl.py
    │   ├── exp04_qwen35.py
    │   ├── requirements-qwen3vl.txt
    │   └── requirements-qwen35.txt
    ├── tests/
    │   ├── test_exp04_qwen3vl.py
    │   └── test_exp04_qwen35.py
    ├── data/
    │   ├── metadata.json
    │   └── images/                   # 48 saved CLEVR-style PNG scenes
    ├── experiments/
    │   └── 04-three-operand-operator-controls/
    │       └── README.md            # methodology and full controls
    └── results/                     # curated historical outputs; commit these
        ├── qwen3-vl-4b/
        └── qwen3.5-4b/
```

**Python scripts are the supported execution interface.** `notebooks/` and newly generated `runs/` are excluded by the **single root `.gitignore`**. A clone therefore needs neither Google Drive nor Colab; it includes the scripts, metadata, images, and curated outputs. Model weights download on first inference run. Qwen inference requires appropriate hardware (the 4-bit setup expects a CUDA-enabled GPU); CPU-only validation does not.

## Reproduce on your machine or a GPU server

Run the following from **`three-operand-experiments/`** after cloning this repository. Use **separate Python environments** for the two models because the required Transformers versions differ.

**CPU-only dataset and sampling validation** (no model download or GPU):

```bash
python -m pip install numpy
python code/exp04_qwen35.py --validate-only
python code/exp04_qwen3vl.py --validate-only
```

These checks should print **48 images, 24 rule/scene sets, seed 2026**.

**Qwen3.5-4B** — uses native Hugging Face processing, with Transformers **5.x**:

```bash
python -m pip install -r code/requirements-qwen35.txt
python code/exp04_qwen35.py --label-sanity --smoke-test
python code/exp04_qwen35.py --experiment main_72
```

**Qwen3-VL-4B-Instruct** — uses Transformers **4.57.6**, plus the **separately obtained** original *looking-beyond-attention* repository by Roshan and collaborators:

```bash
python -m pip install -r code/requirements-qwen3vl.txt
python code/exp04_qwen3vl.py --upstream /path/to/original/looking-beyond-attention --smoke-test
python code/exp04_qwen3vl.py --upstream /path/to/original/looking-beyond-attention --experiment main_72
```

The `--upstream` argument accepts the upstream checkout, its `code/` directory, or a ZIP archive containing `code/vlm/runner.py` and `code/experiments/readout.py`. Upstream code is **not** redistributed here; consult the original project's license and permissions.

**Choose an experiment:** Use `--experiment` with `main_72`, `text_only`, `captioned`, `position_order`, `label_assignment`, `iid_positive`, `ablation`, `perception`, or `gates`. Qwen3-VL additionally supports `discovery_open` and `discovery_guided`; the Qwen3.5 discovery runs were inconclusive due to truncated final responses and are not included in the maintained runner. Example:

```bash
python code/exp04_qwen35.py --experiment ablation
```

**Override input/output paths** with `--data-dir PATH` and `--out PATH`. By default, scripts find `data/` relative to their own source file and save fresh runs to `runs/<model>/<experiment>/`. These runs are resumable by trial ID; archived `results/` files remain unchanged. Each condition saves `config.json`, `runs.jsonl` and `summary.json`.

**Run regression checks** (CPU-only, using mocked model predictions):

```bash
python -m unittest discover -s tests -v
```

## Reproducibility status and limitations

- **Qwen3.5 Python runner:** A real Colab T4 test successfully scored deliberately requested X/Y/Z labels, ran one ten-image smoke test, and completed 72 presentations, yielding **31.9% accuracy**. Matching aggregate accuracy to an earlier run does not by itself establish that per-trial logits are identical.
- **Qwen3-VL Python runner:** Passed CPU-only conversion tests; **the converted script still requires a fresh GPU smoke test**. Its earlier experiments were executed and are preserved in `results/`.
- Both conversions passed **10 combined CPU-only unit tests**. These tests do not establish GPU correctness.
- The 72 presentations reuse 24 scene sets; repeated layouts are not independent observations. X/Y/Z next-token probabilities are normalized over those three tokens only, not the entire output vocabulary. Image preprocessing also differs across the two model wrappers.

## Attribution

This is a follow-up to the operator-control experiments in **Roshan and collaborators' original _looking-beyond-attention_ research code**. Roshan's Qwen3-VL wrapper and readout remain an external dependency, not code claimed as original here. Any upstream redistribution must respect its licensing and permissions.

**Research in progress.** Findings refer only to the experiments and saved outputs supplied here.
