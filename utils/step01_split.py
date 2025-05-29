# step01_split.py
"""
Module: step01_split.py

Splits a BGR video frame into left and right halves, corrects color, and saves as PNGs.
"""
import cv2
from PIL import Image
from pathlib import Path

def split_frame(frame, workdir: Path, idx: int):
    """
    Splits a BGR frame into left and right halves, converts to RGB,
    and saves with a 01 step filename.

    Args:
        frame (np.ndarray): BGR image frame from OpenCV.
        workdir (Path): Directory to save outputs.
        idx (int): Frame index for naming.

    Returns:
        tuple: (left_path, right_path)
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    w, h = img.size

    # Split top (left camera) and bottom (right camera)
    left = img.crop((0, 0, w, h // 2))
    right = img.crop((0, h // 2, w, h))

    # Save outputs
    out_l = workdir / f"frame{idx:06d}_01_left.png"
    out_r = workdir / f"frame{idx:06d}_01_right.png"
    left.save(out_l)
    right.save(out_r)
    print(f"01 - Saved split frames: {out_l}, {out_r}")

    return out_l, out_r

