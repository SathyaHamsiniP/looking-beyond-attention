# Boolean-rule experiments — Qwen3.5-4B

We started with three Boolean rules and tested Qwen3.5-4B using operator controls, probing, activation patching and LoRA. We then extended the experiments to seven rules to examine what changes when rule identification becomes more difficult.

Each question contains images that satisfy one rule. A, B and C mean green cube, blue sphere and red cylinder. The model must choose the generating rule from a list. We ask both whether it answers correctly and what its internal representations tell us. Accurate answers or decodable labels alone do not establish Boolean reasoning.

## Experiments and results

| Experiment | Three rules | Seven rules |
|---|---|---|
| Operator controls | Three generating rules with three or seven choices; separate 10/20-image comparison | [07](experiments/07-seven-rule-operator-controls/README.md): seven rules and choices; 49 validation sets × seven rotations; **48/343 correct** |
| Probing | 90 train / 30 earlier test sets; one diagnostic or ten images; max pixels 301056 | [09](experiments/09-seven-rule-probing/README.md): 147 train / 49 validation sets; one ambiguous, five identifying or ten original images; max pixels 100352; **not yet run on GPU** |
| Activation patching | 3,456 saved measurements from the earlier three-rule experiment | [08](experiments/08-seven-rule-activation-patching/README.md): **896 saved measurements**, 14 pairs, rotation 0, 32 layers, two token positions |
| LoRA | Two epochs, 46 updates: validation **29/90 → 30/90**; selected model always answered Y | [06](experiments/06-seven-rule-lora/README.md): three epochs, 111 planned updates; one successful full-training update saved, **no verified full-run adapter or fine-tuned evaluation** |

### Operator controls

We varied the answer choices and image count to test how the task setup affects answers. For the original three rules, the saved summary reports 36.67% with three choices and 14.76% with seven. The seven-rule experiment scored 13.9942% and answered only positions 6 or 7. These use different datasets, so the difference does not isolate the effect of adding rules.

### Probing

Linear probes test whether a layer's representation contains information that predicts the generating rule. In the earlier three-rule experiment, final-layer accuracy was 100% with one diagnostic image and 50% with ten; shuffled-training-label controls scored 20% and 36.67%. Those scores use the earlier test partition.

The new seven-rule experiment uses validation instead. One image is an ambiguous control; five images are selected using the truth tables and uniquely identify the rule; ten images preserve the original saved set. Five-image selection uses knowledge of the rule, so it differs from the earlier one-image diagnostic condition. CPU checks passed, but there are no seven-rule GPU probing results yet.

### Activation patching

We replace a decoder representation with one from another example and measure changes in answer probabilities. Seven-rule effects vary across layers: different-rule final-token effects range from −0.004397 to +0.008391. At layer 32, the donor-true-rule probability rises less for different-rule donors than for same-rule controls. This does not establish causal Boolean reasoning.

### LoRA

The three-rule adapter gained one correct validation answer but always chose Y. The seven-rule run has no completed fine-tuned evaluation available. A notebook report of 116/343 after LoRA is unverified without its underlying predictions and model. Check remaining backups before considering a new full run; smoke-test weights are not a final model.

## Limitations

The experiments differ in datasets, input sizes, answer choices, image preprocessing, training budgets and evaluation partitions. They are not matched estimates of a three-versus-seven rule effect. Sets reuse images within a partition, and option rotations of a set are repeated measurements. New seven-rule work leaves test untouched. Chance accuracy is 1/3 or 1/7 according to the number of answer choices.

## Results

The [result index](results/README.md) keeps the two rule families and four methods separate. Original results remain in place. Imported files preserve their original bytes; `provenance.json` records source locations, sizes, SHA-256 hashes and numerical checks. Model adapters, optimizer state, images and feature arrays stay outside GitHub. The [import script](import_results.py) rejects mismatched models and conflicting files:

```bash
python benchmark/import_results.py --help
```

## How to run

Start with [09's Colab instructions](experiments/09-seven-rule-probing/README.md) for shared inputs, pinned packages, CPU checks, extraction and probe fitting. Outputs are saved under `/content` and important summaries print before upload. The [uploader](iitm_drive.py) checks `24f2005474@ds.study.iitm.ac.in` and folder `15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR` before writing; it never falls back to personal Drive.

```bash
PYTHONPATH=benchmark python -m unittest discover -s benchmark -p 'test_*.py'
```

Earlier scripts remain available: [three-rule probing](probing.py), [10/20 controls](pilot_10_vs_20.py), [three/seven choices](compare_3_vs_7.py). [Old experiment notes](archive/old-experiment-notes.md) preserve their instructions. Reuse the saved ten-image sets and splits for all new seven-rule work.
