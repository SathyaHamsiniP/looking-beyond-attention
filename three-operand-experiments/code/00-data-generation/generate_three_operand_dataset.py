#!/usr/bin/env python3
"""Render a balanced CLEVR-style Boolean scene dataset with Roshan's renderer.

The renderer is an external dependency: pass the directory that contains
code/clevr_logic (or clevr_logic itself as a sibling directory).

Example:
    python generate_three_operand_dataset.py \
      --renderer-root /content/roshan/looking-beyond-attention-main \
      --output-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000' \
      --num-images 1000 --seed 42

Existing output images are checked and reused so interrupted Colab runs can resume.
The original 48-image dataset is never modified unless you explicitly point the
output at that directory (which is rejected if it has older metadata).
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import random
import sys
from pathlib import Path

from PIL import Image

ATOMS = ("green_cube", "blue_sphere", "red_cylinder")
RULE_EXPRESSIONS = {
    "R1": "(A AND B) OR C",
    "R2": "(A AND C) OR B",
    "R3": "(B AND C) OR A",
}
GENERATOR_VERSION = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer-root", type=Path, required=True,
        help="Path to the reference repo containing code/clevr_logic/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New dataset directory (images/ and metadata.json are placed here).",
    )
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction", type=float, default=0.8,
        help="Fraction of each Boolean configuration allocated to train (default: 0.8).",
    )
    parser.add_argument("--samples", type=int, default=2, help="Renderer samples per pixel.")
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Write metadata after this many images (default: 25).",
    )
    return parser.parse_args()


def load_renderer(root: Path):
    root = root.expanduser().resolve()
    if (root / "code" / "clevr_logic").is_dir():
        code_dir = root / "code"
    elif (root / "clevr_logic").is_dir():
        code_dir = root
    else:
        raise SystemExit(
            f"Cannot find clevr_logic in {root}. Supply --renderer-root pointing "
            "to the extracted reference repository containing code/clevr_logic/."
        )
    sys.path.insert(0, str(code_dir))
    colorshape = importlib.import_module("clevr_logic.colorshape")
    raytracer = importlib.import_module("clevr_logic.raytracer")
    scene = importlib.import_module("clevr_logic.scene")
    return colorshape, raytracer, scene


def labels(bits: tuple[int, int, int]) -> dict[str, int]:
    a, b, c = map(bool, bits)
    return {
        "R1": int((a and b) or c),
        "R2": int((a and c) or b),
        "R3": int((b and c) or a),
    }


def write_metadata(path: Path, metadata: dict, records: list[dict]) -> None:
    snapshot = {**metadata, "completed_images": len(records), "records": records}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_saved_image(path: Path, size: tuple[int, int]) -> None:
    try:
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"incorrect image dimensions {image.size}, expected {size}")
            image.verify()
    except Exception as exc:
        raise SystemExit(f"Invalid existing image {path}: {exc}") from exc


def main() -> None:
    args = arguments()
    if args.num_images < 16 or args.num_images % 8:
        raise SystemExit("--num-images must be a multiple of 8 and at least 16.")
    if not 0 < args.train_fraction < 1:
        raise SystemExit("--train-fraction must be strictly between 0 and 1.")
    if args.samples < 1 or args.checkpoint_every < 1:
        raise SystemExit("--samples and --checkpoint-every must be positive.")

    per_type = args.num_images // 8
    train_per_type = int(per_type * args.train_fraction)
    if not 1 <= train_per_type < per_type:
        raise SystemExit("Train/test split would be empty; adjust --train-fraction.")

    out_dir = args.output_dir.expanduser().resolve()
    image_dir = out_dir / "images"
    meta_path = out_dir / "metadata.json"
    # Refuse to touch unknown files or a pre-existing legacy image dataset.
    if out_dir.exists() and not meta_path.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"Nonempty directory without this generator's metadata: {out_dir}. "
            "Choose a new --output-dir to protect the existing dataset."
        )

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "requested_images": args.num_images,
        "variations_per_type": per_type,
        "train_per_type": train_per_type,
        "train_fraction": args.train_fraction,
        "samples": args.samples,
        "width": 384,
        "height": 256,
        "atoms": list(ATOMS),
        "rules": RULE_EXPRESSIONS,
        "split_definition": "train = variation < train_per_type; test = remaining variations",
    }
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, expected in metadata.items():
            if saved.get(key) != expected:
                raise SystemExit(
                    f"Existing metadata mismatch for '{key}' ({saved.get(key)!r} vs "
                    f"{expected!r}). Choose a different --output-dir."
                )

    CS, raytracer, scene = load_renderer(args.renderer_root)
    target_objects = [CS.split_atom(atom) for atom in ATOMS]
    distractor_pool = [
        (color, shape)
        for color in raytracer.COLORS
        for shape in raytracer.SHAPES
        if (color, shape) not in target_objects
    ]
    if len(distractor_pool) < 4:
        raise SystemExit("Renderer does not supply enough non-target distractors.")

    camera = raytracer.Camera(width=384, height=256)
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    created, reused = 0, 0
    if not meta_path.exists():
        write_metadata(meta_path, metadata, records)

    for bits in itertools.product((0, 1), repeat=3):
        scene_id = "".join(map(str, bits))
        for variation in range(per_type):
            # Independent per-image seeds make resumed runs deterministic even
            # when earlier images are skipped.
            rng = random.Random(f"{args.seed}:{scene_id}:{variation}")
            objects = [obj for obj, present in zip(target_objects, bits) if present]
            objects.extend(rng.sample(distractor_pool, 4 - len(objects)))
            rng.shuffle(objects)
            filename = f"scene_{scene_id}_{variation:03d}.png"
            destination = image_dir / filename
            if destination.exists():
                check_saved_image(destination, (384, 256))
                reused += 1
            else:
                placed = scene.place_objects(objects, rng, camera)
                pixels = raytracer.render(
                    placed,
                    raytracer.jitter_camera(rng, camera),
                    samples=args.samples,
                    lights=raytracer.jitter_lights(rng),
                )
                # Save to a temporary PNG first so interrupted writes cannot
                # leave a partially written destination that looks complete.
                temp = destination.with_name(destination.stem + ".tmp.png")
                Image.fromarray(pixels).save(temp)
                temp.replace(destination)
                created += 1

            records.append({
                "filename": filename,
                "bits": list(bits),
                "variation": variation,
                "split": "train" if variation < train_per_type else "test",
                "objects": [list(obj) for obj in objects],
                "rule_labels": labels(bits),
            })
            if len(records) % args.checkpoint_every == 0:
                write_metadata(meta_path, metadata, records)
        print(f"Finished {scene_id}: {per_type} images ({len(records)}/{args.num_images})", flush=True)
        write_metadata(meta_path, metadata, records)

    print(
        f"Done: {len(records)} images; created {created}, reused {reused}.\n"
        f"Train: {8 * train_per_type}; test: {8 * (per_type - train_per_type)}.\n"
        f"Images: {image_dir}\nMetadata: {meta_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
