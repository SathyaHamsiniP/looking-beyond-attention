# Inspect the completed baseline

Read the [Experiment 03 guide](../README.md) first. It explains the inputs, patching, measurements and limits.

## What is available here?

The [analysis script](../../../code/03-activation-patching/analyze_baseline.py) reads the original results without loading Qwen. The uploaded ZIP contained the completed run and a separate smoke test. Only the completed `pairs2_rot3_prev1` run is analysed here.

- [raw/](raw/): byte-for-byte copies of the five original files. Google Drive was not modified.
- [derived/analysis.md](derived/analysis.md): findings and limitations checked against the raw CSV.
- [derived/](derived/): input manifest, metrics, plots, examples, provenance and local HTML explorer.
- [explorer.html](derived/explorer.html): download this file and open it in a browser. GitHub displays HTML source rather than running the explorer. Select the local dataset images folder with its file picker to display images; no images are embedded or committed.

The checks confirm 3,456 trials, all 18 pairs, three rotations, 32 layers and two positions. Ordered inputs match the saved dataset fingerprint. The original layer averages and the layer-6 peak/control result agree with recalculation.

Required original files: `config.json`, `patches.csv`, `layer_effects.csv`, and `summary.json`. Keep `layer_effects.png` with the original archive too; the script recomputes its own plots from the CSV. Do not edit or move the Drive originals.

## Run the analysis

From the repository root, with the existing NumPy/Matplotlib dependencies installed:

```bash
python three-operand-experiments/code/03-activation-patching/analyze_baseline.py \
  --source /path/to/copied/pairs2_rot3_prev1 \
  --data-dir /path/to/data-1000 \
  --output /path/to/new/baseline-analysis
```

Choose a new output folder outside both source directories. Existing output folders are rejected. The script checks the pair plan, image fingerprint, trial keys, flags, probability changes and aggregate means. It never runs inference.

It writes ordered input manifests, derived metrics, original predictions, verified counts, representative trial records, plots, an `analysis.md` and a local `explorer.html`. Open the HTML file directly in a browser on the same machine as the dataset. Images use local file links to the dataset; the report does not embed or duplicate all 1,000 pictures. Browser settings may restrict these local image links. Trial text still remains readable. No web server is needed.

Reconstructed probabilities are explicitly identified in `examples.json`. Unknown same-rule probabilities remain null. The explorer shows the saved fields, images, bit labels and full prompt for the selected pair, rotation, layer and position. Its bit labels are inspection aids, not inputs supplied to Qwen.

`provenance.json` records source checksums, the run's dataset fingerprint, analysis script checksum, current code commit and command arguments. Preserve that file with each generated report. The original baseline config does not itself record a Git commit; the analysis commit must not be mistaken for the original run's code revision.

## Reproducing the published files

Use `raw/` as `--source` and a fresh output folder. The script refuses existing output folders to avoid replacing earlier reports. Generated provenance records the exact command and script checksum. Absolute dataset paths reflect the analysis machine; use the explorer file picker on another computer.

The code commit in provenance identifies the checkout used for analysis, not the original GPU run. The working tree contained the new script when these artifacts were generated; its SHA256 identifies the exact script. The original config did not record its run commit. No code revision was invented for it.

All source checksums were unchanged after processing. Raw copies were also checked against the uploaded archive. The generated plots are measurements from these files, not illustrative model results. The source `layer_effects.png` remains separate from the new plots.
