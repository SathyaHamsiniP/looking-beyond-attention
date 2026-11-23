"""Rebuild the portfolio figure from the two saved prediction files."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
results = here.parent / "benchmark/results/seven-rules/qwen3.5-4b"
fig, (accuracy, position) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
for i, (name, folder, color) in enumerate([
    ("Pretrained", "operator-controls", "#516580"),
    ("LoRA", "lora/finetuned-validation", "#008477"),
]):
    rows = [json.loads(line) for line in (results / folder / "predictions.jsonl").read_text().splitlines()]
    percent = 100 * sum(row["correct"] for row in rows) / len(rows)
    bars = accuracy.bar(name, percent, color=color, width=0.55)
    accuracy.bar_label(bars, labels=[f"{percent:.2f}%"], padding=3)
    labels = [int(row["predicted_label"]) for row in rows]
    positions = range(1, 8)
    position.bar([x + (i - 0.5) * 0.36 for x in positions], [labels.count(x) for x in positions],
                 width=0.36, label=name, color=color)
accuracy.set(title="Correct answers", ylabel="Validation accuracy (%)", ylim=(0, 40))
position.set(title="Where Qwen placed its answer", xlabel="Option position", ylabel="Presentations", xticks=range(1, 8))
position.legend(frameon=False)
fig.savefig(here / "lora-validation.png", dpi=170)
