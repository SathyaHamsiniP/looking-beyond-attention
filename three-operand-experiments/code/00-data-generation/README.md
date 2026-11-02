# 00 — Generate logic images

This script makes 1,000 images with green cubes, blue spheres, red cylinders, and other objects.

It uses the image renderer from [Roshan's repository](https://github.com/RoshanYSingh23/looking-beyond-attention).

## Run on Google Colab

Run these cells in order. A GPU is not needed.

**1. Connect Google Drive.**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**2. Get the code.**

```python
!git clone https://github.com/SathyaHamsiniP/looking-beyond-attention.git /content/looking-beyond-attention
!mkdir -p /content/roshan
!git clone https://github.com/RoshanYSingh23/looking-beyond-attention.git /content/roshan/looking-beyond-attention-main
!git -C /content/roshan/looking-beyond-attention-main checkout 71ee6120e129bd466f5cc99745a20060e61403b6
!pip install -q numpy pillow
```

If you already cloned the repositories in this Colab session, skip those clone commands.

**3. Generate the images.**

```python
!python /content/looking-beyond-attention/three-operand-experiments/code/00-data-generation/generate_three_operand_dataset.py \
  --renderer-root /content/roshan/looking-beyond-attention-main \
  --output-dir "/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/data-1000" \
  --num-images 1000 --seed 42
```

## What gets saved?

Images and `metadata.json` are saved to `data-1000/` on Google Drive.

- 1,000 images: 125 for each of the 8 A/B/C combinations.
- 800 training images and 200 test images.
- `metadata.json` lists the objects, labels, and train/test group for each image.

The old 48 images stay unchanged. Run the same command again if Colab stops. Finished images will be reused.

**Note:** The current probing scripts still use the old 48-image setup. Update them before using this new dataset. The images are **not** uploaded to GitHub automatically.
