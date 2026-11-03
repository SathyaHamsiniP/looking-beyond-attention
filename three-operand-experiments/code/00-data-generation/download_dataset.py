#!/usr/bin/env python3
"""Download and check the 1,000-image CLEVR dataset from GitHub Releases."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

URL = (
    "https://github.com/SathyaHamsiniP/looking-beyond-attention/"
    "releases/download/dataset-v1/clevr-logic-1000.zip"
)
SHA256 = "86b91e44d617cb7d9ffc893c4459e5e5eb621841eb98f36f166d7e67dfd2f030"


def check_dataset(root: Path) -> None:
    metadata = root / "metadata.json"
    image_dir = root / "images"
    if not metadata.is_file() or not image_dir.is_dir():
        raise ValueError(f"Missing images/ or metadata.json in {root}")
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    records = saved["records"]
    names = [r["filename"] for r in records]
    bits = Counter(tuple(r["bits"]) for r in records)
    splits = Counter(r["split"] for r in records)
    images = list(image_dir.glob("*.png"))
    if (
        len(records) != 1000
        or len(names) != len(set(names))
        or len(images) != 1000
        or bits != Counter({b: 125 for b in (
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))})
        or splits != Counter({"train": 800, "test": 200})
        or {p.name for p in images} != set(names)
    ):
        raise ValueError("Dataset check failed: image counts, splits or labels differ.")
    print("Dataset checked: 1,000 images; 800 train and 200 test.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "data-1000",
        help="Where to save the dataset.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Check the existing dataset without downloading anything.",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        check_dataset(output)
        print(f"Already present: {output}")
        return
    if args.check_only:
        parser.error(f"Dataset not found: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clevr_dataset_", dir=output.parent) as temp:
        work = Path(temp)
        archive = work / "dataset.zip"
        print("Downloading the 1,000 images...", flush=True)
        request = Request(URL, headers={"User-Agent": "clevr-logic-dataset-downloader"})
        digest = hashlib.sha256()
        with urlopen(request, timeout=120) as source, archive.open("wb") as dest:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                dest.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError("The downloaded ZIP does not match the published file.")

        unpacked = work / "unpacked"
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            if not names or any(
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or not Path(name).parts
                or Path(name).parts[0] != "data-1000"
                for name in names
            ):
                raise ValueError("Unexpected paths in the ZIP.")
            zipped.extractall(unpacked)

        dataset = unpacked / "data-1000"
        check_dataset(dataset)
        shutil.move(str(dataset), str(output))
        print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
