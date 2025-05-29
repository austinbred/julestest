# step05_canvas.py
"""
Module: step05_canvas.py

Stitches two rig-space images into a panorama canvas using a known pixel shift.
Logs image sizes, shift, and canvas parameters.
"""
import cv2
import numpy as np
from pathlib import Path


def compose_canvas(left_bgr: np.ndarray,
                   right_bgr: np.ndarray,
                   shift_px: int,
                   workdir: Path,
                   idx: int) -> np.ndarray:
    """
    Stitches left and right BGR images side-by-side on a dynamically sized canvas.

    Args:
        left_bgr  (np.ndarray): Left rig-space image in BGR.
        right_bgr (np.ndarray): Right rig-space image in BGR.
        shift_px   (int): Horizontal pixel offset for right image relative to left.
        workdir    (Path): Directory to save the stitched output.
        idx        (int): Frame index for naming.

    Returns:
        np.ndarray: The stitched panorama in BGR.
    """
    # Input sizes
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + shift_px + w2
    print(f"05 - Left size: width={w1}, height={h1}")
    print(f"05 - Right size: width={w2}, height={h2}")
    print(f"05 - Using shift_px: {shift_px}")
    print(f"05 - Canvas size: width={canvas_w}, height={canvas_h}")

    # Create black canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=left_bgr.dtype)

    # Paste left at (0,0)
    canvas[0:h1, 0:w1] = left_bgr
    # Paste right at (shift)
    x_off = w1 + shift_px
    canvas[0:h2, x_off:x_off + w2] = right_bgr
    print(f"05 - Pasted right at x offset {x_off}")

    # Optional seam blending (uncomment if needed):
    # blend_width = 50
    # for i in range(blend_width):
    #     alpha = i / (blend_width - 1)
    #     x = w1 + shift_px - blend_width//2 + i
    #     canvas[:, x] = (
    #         (1-alpha) * canvas[:, x] + alpha * right_bgr[:, i]
    #     ).astype(canvas.dtype)

    # Save stitched canvas
    out_path = workdir / f"frame{idx:06d}_05_stitched.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"05 - Saved stitched canvas: {out_path}")
    return canvas

