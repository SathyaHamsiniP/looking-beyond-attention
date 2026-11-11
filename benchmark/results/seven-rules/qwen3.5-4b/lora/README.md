# Seven-rule LoRA

**No verified completed full-run adapter or fine-tuned predictions.** [Saved file inventory](artifact_audit.json) lists the available files. Partial diagnostics show one successful full-training update; [pretrained metrics](pretrained-validation/metrics.json) match the 343 predictions stored in [operator controls](../operator-controls/predictions.jsonl). Smoke weights are excluded. The notebook 116/343 claim is unverified. See [06](../../../../experiments/06-seven-rule-lora/README.md) for conditional recovery and backup instructions.

## Publishing a completed run

When the final archive is verified, this folder should contain only the completed run's small training/evaluation records and any existing comparison plots. Keep the source ZIP, adapter weights, optimizer state, temporary checkpoints, smoke-test outputs and datasets outside GitHub. The pretrained evaluation is already represented in [operator controls](../operator-controls/README.md). The earlier partial diagnostics and file audit can be removed from this active results folder after they are superseded, while their history remains in Git.

**The full-run result ZIP has not yet been verified for this README.** The status above describes the last checked GitHub artifacts; replace it with the measured final result and links only after the file-level checks pass.
