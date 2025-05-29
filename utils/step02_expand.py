"""
Module: step02_expand.py

Expands each half to a larger blank canvas, centering the original image,
and saves with updated step-wise filenames.
"""
from PIL import Image
from pathlib import Path

def expand_image(img_path: Path, workdir: Path, scale: float = 2.0) -> Path:
    """
    Loads img_path (e.g., frame000001_01_left.png), expands it,
    and saves as frame000001_02_left_exp.png in workdir.
    Returns the Path to the expanded image.
    """
    img = Image.open(img_path)
    w, h = img.size
    exp_w, exp_h = int(w * scale), int(h * scale)
    ox, oy = (exp_w - w) // 2, (exp_h - h) // 2
    canvas = Image.new(img.mode, (exp_w, exp_h))
    canvas.paste(img, (ox, oy))

    # Parse stem: [frame, step1, side]
    parts = img_path.stem.split('_')  # e.g. ['frame000001','01','left']
    frame = parts[0]
    side = parts[2]
    out_name = f"{frame}_02_{side}_exp.png"
    out_path = workdir / out_name
    canvas.save(out_path)
    print(f"02 - Saved expanded image: {out_path}")
    return out_path
