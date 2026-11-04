"""Content-addressed, configuration-checked new runs and append-only records."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def provenance():
    def git(*args):
        return (
            subprocess.check_output(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[3],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

    versions = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "numpy",
        "scikit-learn",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        git_commit=git("rev-parse", "HEAD"),
        git_diff_sha256=hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        python=platform.python_version(),
        packages=versions,
    )


def file_fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(manifest, data_dir):
    names = sorted({o["path"] for s in manifest["sets"] for o in s["observations"]})
    return digest({name: file_fingerprint(Path(data_dir) / name) for name in names})


def prepare_run(root, experiment, manifest, config):
    from .results_paths import run_path

    config = json.loads(
        json.dumps(
            {**config, "manifest_sha256": digest(manifest), "provenance": provenance()}
        )
    )
    out = run_path(root, experiment, manifest["task_version"], digest(config)[:16])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Run configuration collision")
    if not path.exists():
        atomic_json(path, config)
    atomic_json(out / "task_manifest.json", manifest)
    atomic_json(out / "split_manifest.json", manifest["split_manifest"])
    print("Run directory:", out, flush=True)
    return out


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row["key"] in self.rows:
                    raise ValueError("Duplicate result key")
                self.rows[row["key"]] = row

    def add(self, key, row):
        if key in self.rows:
            raise ValueError("Result already saved")
        row = {**row, "key": key}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows[key] = row
