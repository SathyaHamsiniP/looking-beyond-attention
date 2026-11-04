"""Layer-wise rule probes fitted only to manifest training sets."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_concepts import capture
from shared.data_loader import load_manifest, option_labels, prompt
from shared.model_qwen35 import load, model_metadata
from shared.run_utils import atomic_json, dataset_fingerprint, prepare_run


def fit_probes(features, items, partition, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = [i for i, s in enumerate(items) if s["partition"] == "train"]
    test = [i for i, s in enumerate(items) if s["partition"] == partition]
    y = np.array([s["rule"] for s in items])
    counts = Counter(y[train])
    majority = max(counts, key=counts.get)
    rows = []
    for layer in range(features.shape[1]):
        for pool, name in enumerate(("last", "text_mean", "visual_mean")):
            for shuffled in (False, True):
                labels = y[train].copy()
                if shuffled:
                    np.random.default_rng(seed).shuffle(labels)
                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, random_state=seed),
                )
                probe.fit(features[train, layer, pool], labels)
                prediction = probe.predict(features[test, layer, pool])
                rows.append(
                    dict(
                        layer=layer + 1,
                        pool=name,
                        control="shuffled" if shuffled else "measured",
                        accuracy=float(np.mean(prediction == y[test])),
                        train_size=len(train),
                        test_size=len(test),
                        majority_baseline=float(np.mean(y[test] == majority)),
                        chance=1 / len(counts),
                        train_counts=json.dumps(dict(counts)),
                        test_counts=json.dumps(dict(Counter(y[test]))),
                    )
                )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("rules", "concepts"), default="rules")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--partition", choices=("validation", "test"), default="validation")
    p.add_argument("--precision", default="4bit")
    p.add_argument("--revision", default="main")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    m = load_manifest(a.manifest, a.data_dir)
    items = [s for s in m["sets"] if s["partition"] in ("train", a.partition)]
    if a.kind == "concepts":
        records = json.loads((a.data_dir / "metadata.json").read_text())["records"]
        items = [
            dict(
                set_id=Path(r["filename"]).stem,
                partition=m["split_manifest"][r["filename"]],
                rule="".join(map(str, r["bits"])),
                observations=[dict(path="images/" + r["filename"])],
            )
            for r in records
            if m["split_manifest"][r["filename"]] in ("train", a.partition)
        ]
    if a.validate_only:
        print(Counter(s["partition"] for s in items))
        return
    cfg = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("results_root", "validate_only")
    }
    cfg["data_fingerprint"] = dataset_fingerprint(m, a.data_dir)
    adapter = load(a.precision, a.revision)
    cfg["model"] = model_metadata(adapter, option_labels(len(m["rules"])))
    out = prepare_run(a.results_root, 2, m, cfg)
    atomic_json(out / "model.json", cfg["model"])
    if a.smoke_test:
        items = items[:2]
    features = []
    for item in items:
        cache = out / (item["set_id"] + ".npz")
        if not cache.exists():
            if a.kind == "concepts":
                paths = [str(a.data_dir / item["observations"][0]["path"])]
                text = "Describe the objects in this image."
            else:
                paths, text, _ = prompt(item, m, a.data_dir)
            # Remove choices to avoid probing answer-position mappings or expression tokens.
            if a.kind == "rules":
                text = (
                    text.split("\n" + ("X" if len(m["rules"]) == 3 else "D") + ":")[0]
                    + "\nDescribe the hidden rule."
                )
            values = capture(adapter, paths, text, "qwen3_5")
            temp = cache.with_suffix(".tmp.npz")
            np.savez_compressed(temp, features=values)
            temp.replace(cache)
        with np.load(cache) as f:
            features.append(f["features"])
    if a.smoke_test:
        print("Two extraction forwards completed; no probe fitted")
        return
    rows = fit_probes(np.stack(features), items, a.partition, a.seed)
    with (out / "probe_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib.pyplot as plt

    for pool in ("last", "text_mean", "visual_mean"):
        selected = [r for r in rows if r["pool"] == pool and r["control"] == "measured"]
        plt.plot(
            [r["layer"] for r in selected],
            [r["accuracy"] for r in selected],
            label=pool,
        )
    plt.legend()
    plt.xlabel("Decoder layer")
    plt.ylabel("Held-out rule probe accuracy")
    plt.tight_layout()
    plt.savefig(out / "probe_accuracy.png")
    plt.close()
    atomic_json(
        out / "summary.json",
        dict(
            partition=a.partition,
            interpretation="Decodable information; not evidence of causal use. Overlapping sets are not independent.",
        ),
    )


if __name__ == "__main__":
    main()
