# Looking Beyond Attention

**Experiments on logical reasoning and evidence use in vision-language models.**

This project investigates a question that cannot be answered by inspecting attention weights alone: **does a model use the relevant evidence to perform a logical computation, or does it arrive at an answer through other cues and biases?**

The work uses controlled visual and text-based tasks to separate several abilities that are easy to conflate:

- **Perception:** recognizing the concepts present in an image.
- **Boolean execution:** evaluating an explicitly supplied logical expression.
- **Rule identification:** inferring a hidden rule from multiple examples.
- **Evidence sensitivity:** changing predictions appropriately when decisive evidence is added or removed.
- **Presentation robustness:** preserving the inferred rule when answer labels or option positions change.

Successful concept recognition or individual Boolean calculations do not, by themselves, establish successful multi-example rule inference. Likewise, attention patterns alone are not causal evidence of how a model computed its answer.

## Experiments

| Study | Focus | Materials |
| --- | --- | --- |
| **Three-operand operator controls** | Can Qwen vision-language models identify a hidden three-concept Boolean rule from CLEVR-style scenes? How do visual input, text controls, label order, and other interventions affect their choices? | [Experiment folder](three-operand-experiments/) · [Methodology and results](three-operand-experiments/experiments/04-three-operand-operator-controls/README.md) |

The repository is organized to accommodate further experiments. Only the study above is documented here at present; additional experiments should be added when their code, data, and results are ready.

## Current findings: three-operand rule identification

The first study compares **Qwen3-VL-4B-Instruct** with **Qwen3.5-4B**. Models are shown ten positive CLEVR-style scenes and asked to choose among three candidate rules:

- `R1 = (A AND B) OR C`
- `R2 = (A AND C) OR B`
- `R3 = (B AND C) OR A`

Here, `A` is a green cube, `B` a blue sphere, and `C` a red cylinder. The main evaluation includes **72 presentations per model**: three generating rules, eight scene sets per rule, and three cyclic answer-label assignments.

| Evaluation | Qwen3-VL-4B | Qwen3.5-4B |
| --- | ---: | ---: |
| Hidden-rule identification (72 presentations) | 33.3% | 31.9% |
| Both stages of an explicit Boolean gate calculation correct (24 cases) | 79.2% | 95.8% |
| Concept recognition (8 held-out images) | 8/8 | 8/8 |

The three-choice chance baseline is **33.3%**. Both models performed near that baseline on the hidden-rule task despite strong concept recognition, and their selections exhibited substantial option-position bias. The gate results suggest that the difficulty is not simply an inability to evaluate the candidate Boolean expressions when these are given explicitly.

**Interpretation limits:** Repeated option layouts of a given scene set are not independent samples. Candidate next-token probabilities are normalized over the three offered labels, not the full vocabulary. Positive-only datasets must be checked for whether they uniquely identify a rule. The Qwen3.5 open/guided free-text discovery runs were excluded from the cleaned comparison because truncated responses prevented reliable evaluation; unparseable responses must not be treated as proof of failed reasoning. See the experiment README for controls, methodology, and caveats.

## Repository layout

```text
looking-beyond-attention/
├── README.md
└── three-operand-experiments/
    ├── README.md
    ├── notebooks/
    │   └── exp04/
    │       ├── Qwen3_VL_4B.ipynb
    │       └── Qwen3_5_4B.ipynb
    ├── experiments/
    │   └── 04-three-operand-operator-controls/
    │       └── README.md
    ├── data/
    │   ├── metadata.json
    │   └── images/
    └── results/
        ├── qwen3-vl-4b/
        └── qwen3.5-4b/
```

For each retained experiment condition, `config.json` records how it was run, `runs.jsonl` contains trial-level outputs, and `summary.json` records aggregate metrics. These three files are complementary and are kept together for auditability.

## Reproducing a study

Start with the [three-operand experiment README](three-operand-experiments/README.md), then open the relevant notebook in Google Colab with a GPU runtime. The Qwen3-VL and Qwen3.5 notebooks have different Transformers dependencies and should be run in separate environments.

**Current reproducibility constraint:** The notebooks were originally designed for a Google Drive layout under `MyDrive/LogicCBM++/` and expect their experiment data and result directories there. The Qwen3-VL notebook additionally expects access to the original upstream `looking-beyond-attention` source archive for its model wrapper and answer-logit readout. A GitHub clone by itself does not currently supply that archive. Consult the notebook setup cells before running; this is research code, not yet a portable one-command package.

## Relationship to earlier work

The three-operand experiments extend the operator-control methodology in **Roshan and collaborators' _looking-beyond-attention_ research code**, including its CLEVR-style logic tasks and controlled option-scoring approach. This repository documents our follow-up experimental setup and results; it is **not a claim of authorship or a replacement for the upstream project**. Please check the upstream project's licensing/permission terms before copying or redistributing its implementation.

## Status

Research in progress. The available notebooks and recorded results document experiments already performed; ideas discussed for future experiments should not be interpreted as implemented or validated results.
