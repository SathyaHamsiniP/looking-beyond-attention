# 00 — CLEVR-style three-operand dataset generation

This script recreates the *type of scenes* used in the original 48-image Colab pilot, and scales the dataset to 1,000 new images. It uses the same external clevr_logic renderer calls (place_objects, render, jitter_camera, jitter_lights). It is **not** a byte-for-byte reconstruction of the old 48 images: its random seeds are independent per image, allowing safe resume.

## Renderer dependency and attribution

The image renderer comes from [RoshanYSingh23/looking-beyond-attention](https://github.com/RoshanYSingh23/looking-beyond-attention), specifically `code/clevr_logic/{colorshape,raytracer,scene}.py` (and their internal imports). The reference repository is **public**. You do **not** need a privately shared ZIP. Clone it at the commit tested for this setup:

```bash
git clone https://github.com/RoshanYSingh23/looking-beyond-attention.git /content/roshan/looking-beyond-attention-main
git -C /content/roshan/looking-beyond-attention-main checkout 71ee6120e129bd466f5cc99745a20060e61403b6
```

This generator imports that code; it does not redistribute the upstream source. A root `LICENSE` file was not found in the reference repository at the time these instructions were written. Check upstream licensing/permission before copying renderer files into another repository.

Install lightweight image-generation dependencies:

```bash
python -m pip install numpy pillow
```

## In Colab

Cell 1 — mount Drive and clone this repo (or run `git pull` if it is already cloned):

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
```

Cell 2 — obtain the reference renderer from its **public repository**:

```python
!git clone https://github.com/RoshanYSingh23/looking-beyond-attention.git /content/roshan/looking-beyond-attention-main
!git -C /content/roshan/looking-beyond-attention-main checkout 71ee6120e129bd466f5cc99745a20060e61403b6
!python -m pip install -q numpy pillow
```

If you already cloned either repository in the same Colab session, skip its clone command. Check that `/content/roshan/looking-beyond-attention-main/code/clevr_logic/` exists.

Cell 3 — render and save to Drive:

    !python /content/looking-beyond-attention/three-operand-experiments/code/00-data-generation/generate_three_operand_dataset.py \
      --renderer-root /content/roshan/looking-beyond-attention-main \
      --output-dir '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000' \
      --num-images 1000 --seed 42

The output is:

    data-1000/
      images/            # 1,000 files: scene_000_000.png, ...
      metadata.json      # per-image Boolean bits, distractors, split and R1/R2/R3 truth values

## Design and safety

- Eight A/B/C configurations x 125 independently randomized images each = 1,000 images.
- Four objects per image: present target objects plus non-target color-shape distractors.
- 384x256 rendering with 2 samples, matching the previous pilot resolution and render setting.
- Default 80/20 split *within each Boolean configuration*: 800 train and 200 test images. Image files do not cross splits.
- The seed is fixed and per-image; resuming with exactly the same command reuses existing valid PNGs and rebuilds metadata. No automatic upload to GitHub takes place.
- Metadata checkpoints are written periodically and at the end of each configuration.
- The generator rejects an unknown nonempty output directory to protect the original images. Do **not** point it at the old data/ directory.

## Before using this in model experiments

The older experiment scripts assume exactly 48 images and split records based on variation < 4 versus variations 4-5. That code must be updated to use the explicit metadata split values in the new dataset before training/evaluation. In particular, build image-disjoint train/test sets and check the rule labels against R1/R2/R3. Generating 1,000 images does not automatically update model experiments.

Inspect sample images for object occlusions and other rendering artifacts before relying on the dataset. Store large generated assets on Google Drive; GitHub hosts this generator and documentation only. For local runs (no Colab or Drive), supply a local --output-dir and a local --renderer-root pointing to your clone of the public reference repository.
