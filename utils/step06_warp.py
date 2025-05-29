# step06_warp.py
"""
Module: step06_warp.py

Applies a bird's-eye perspective warp to the final canvas using
selected corners from JSON, with detailed logging.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def warp_birdseye(canvas, data: dict,
                  workdir: Path, idx: int) -> np.ndarray:
    """
    Uses data['selected_corners']['corners_stacked'] for TL, TR, BR, BL
    to compute perspective transform and output warped image.
    Logs source and destination points, homography, and saves the warped image.

    Args:
        canvas (PIL.Image or np.ndarray): The stitched canvas.
        data (dict): Calibration JSON data including selected_corners.
        workdir (Path): Directory to save warped output.
        idx (int): Frame index for naming.

    Returns:
        np.ndarray: Warped bird's-eye view BGR array.
    """
    # Convert input canvas to BGR numpy array
    if isinstance(canvas, np.ndarray):
        vis = canvas  # already BGR
    else:
        vis = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    h_vis, w_vis = vis.shape[:2]
    print(f"06 - Canvas size for warping: width={w_vis}, height={h_vis}")

    # Extract selected_corners from JSON
    sels = data.get('selected_corners') or data.get('alignment', {}).get('selected_corners', {})
    corner_data = sels.get('corners_stacked', {}) if isinstance(sels, dict) else {}

    # Define expected corner keys in order: TL, TR, BR, BL
    expected = ['fl', 'fr', 'cr', 'cl']
    src = []
    for key in expected:
        entry = corner_data.get(key)
        if entry is None:
            print(f"06 - Warning: missing corner '{key}', skipping.")
            continue
        # Unpack normalized coords
        if isinstance(entry, dict):
            coords = entry.get('stacked') or entry.get('alt')
        else:
            coords = entry
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            print(f"06 - Warning: invalid format for corner '{key}': {coords}")
            continue
        try:
            xnorm, ynorm = float(coords[0]), float(coords[1])
        except Exception:
            print(f"06 - Warning: non-numeric coords for '{key}': {coords}")
            continue
        x = xnorm * w_vis
        y = ynorm * h_vis
        src.append((x, y))
    if len(src) != 4:
        print(f"06 - Warning: need 4 valid corners but got {len(src)}; skipping bird's-eye warp. src: {src}")
        return vis
    src_pts = np.array(src, dtype=np.float32)
    print(f"06 - Source pts (pixel coords): {src_pts.tolist()}")

    # Destination rectangle based on field dimensions
    field_w = float(data.get('field_width', 1.0))
    ppm = w_vis / field_w
    dst_pts = np.array([
        [0, 0],
        [field_w * ppm, 0],
        [field_w * ppm, field_w * ppm],
        [0, field_w * ppm]
    ], dtype=np.float32)
    print(f"06 - Destination pts (pixel coords): {dst_pts.tolist()}")

    # Compute homography
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print(f"06 - Computed homography matrix:\n{H}")

    # Apply warpPerspective
    bev_w, bev_h = int(field_w * ppm), int(field_w * ppm)
    warped = cv2.warpPerspective(vis, H, (bev_w, bev_h))
    print(f"06 - Warped size: width={bev_w}, height={bev_h}")

    # Save bird's-eye view
    out_path = workdir / f"frame{idx:06d}_06_birds_eye.png"
    Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).save(out_path)
    print(f"06 - Saved bird's-eye view: {out_path}")

    return warped

