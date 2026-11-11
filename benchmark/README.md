# Boolean-rule benchmark — Qwen3.5-4B

Can a vision-language model identify the Boolean rule shared by positive images? We compare its answers, frozen internal representations, activation interventions and LoRA adaptation. Accurate answers alone do not establish Boolean reasoning: answer position and diagnostic scenes can supply shortcuts.

| Method | Earlier three-rule work | Seven-rule suite |
|---|---|---|
| Operator controls | Three generating rules; three/seven choices in A1/A2; separate 10/20-image pilot | [07](experiments/07-seven-rule-operator-controls/README.md): seven rules/choices, 49 validation sets × seven rotations; 48/343 correct |
| Frozen probes | 90 train / 30 historical test sets; one diagnostic or ten images; max pixels 301056 | [09](experiments/09-seven-rule-probing/README.md): exact 147 train / 49 validation sets; 1 ambiguous / 5 oracle-identifying / 10 original images; max pixels 100352; **not yet run** |
| Activation patching | Historical three-rule interventions, 3,456 saved rows | [08](experiments/08-seven-rule-activation-patching/README.md): 896 saved measurements; 14 pairs, rotation 0, 32 layers, two positions |
| LoRA | Two epochs, 46 updates: validation 29/90 → 30/90; selected model always Y | [06](experiments/06-seven-rule-lora/README.md): three epochs, 111 planned updates; only one full-training update recovered, no verified selected full-run adapter |

The historical probe's final layer scores are 100% (one diagnostic image) and 50% (ten images), with shuffled-training-label scores 20% and 36.67%. These use the historical test partition and a different image budget, split and identifiability condition. They are not matched estimates of a three-versus-seven rule effect. New seven-rule work never evaluates the test partition. Chance is 1/3 or 1/7 according to the number of choices.

## Results and integrity

[Compact result index](results/README.md) separates generating-rule families and methods. Original locations remain unchanged. Imports contain original bytes, source URLs and SHA-256 receipts; no model adapters, training binaries, source images or feature arrays are committed. [Safe importer](import_results.py) accepts exact local folders and rejects model/family mismatches and conflicting overwrites:

```bash
python benchmark/import_results.py --help
```

[Audit and recovery](lora_recovery.py) never treats smoke weights or an unfinished staging folder as a full checkpoint. The notebook claim of 116/343 after seven-rule LoRA remains unverified; it is not a measured result in this index.

## Reproduction

Use [09's complete Colab instructions](experiments/09-seven-rule-probing/README.md) for shared School Drive inputs, pinned packages, CPU validation, extraction and probe fitting. All new outputs are written under `/content` before [School Drive upload](iitm_drive.py). Authentication verifies `24f2005474@ds.study.iitm.ac.in` and folder `15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR` before uploads. No fallback to personal Drive is allowed. Important summaries also print to the console.

```bash
PYTHONPATH=benchmark python -m unittest test_build_sets test_compare_3_vs_7 test_lora_7rules test_seven_rule_extensions test_benchmark_suite test_saved_suite_results
```

Historical scripts remain available: [three-rule probing](probing.py), [10/20 controls](pilot_10_vs_20.py), [three/seven choices](compare_3_vs_7.py). [Archived protocol notes](archive/legacy-protocols.md) preserve earlier instructions; they do not describe the current execution status. Existing ten-image sets and their saved splits must be reused, never regenerated for this suite.
