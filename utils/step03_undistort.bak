# step03_undistort.py
"""
Module: step03_undistort.py

Applies fisheye undistortion to expanded images and saves
with updated step-wise filenames.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def undistort_fisheye(img_path: Path, K: np.ndarray, D: np.ndarray, workdir: Path) -> Path:
    """
    Loads img_path (e.g. frame000001_02_left_exp.png), undistorts it,
    and saves as frame000001_03_left_ud.png in workdir.
    Returns the Path to the undistorted image.
    """
    # Read and convert to BGR for OpenCV
    arr = cv2.cvtColor(np.array(Image.open(img_path)), cv2.COLOR_RGB2BGR)
    # Undistort (fisheye)
    und = cv2.fisheye.undistortImage(arr, K, D, Knew=K)
    # Convert back to RGB
    und_rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)

    # Parse stem: [frame, '02', side, 'exp']
    parts = img_path.stem.split('_')  # e.g. ['frame000001','02','left','exp']
    frame = parts[0]
    side = parts[2]
    out_name = f"{frame}_03_{side}_ud.png"
    out_path = workdir / out_name
    Image.fromarray(und_rgb).save(out_path)
    print(f"03 - Saved undistorted image: {out_path}")
    return out_path
