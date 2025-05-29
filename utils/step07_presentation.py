# step07_presentation.py
"""
Module: step07_presentation.py

Transforms the rig-space panorama into the presentation camera view
using the camera2presentation matrix and presentation intrinsics.
Logs all matrices and parameters and saves the final output.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def compute_homography_presentation(K_src: np.ndarray,
                                    cam2pres: np.ndarray,
                                    K_dst: np.ndarray,
                                    fallback_size: tuple) -> tuple:
    """
    Compute a 3x3 homography H for rig->presentation ignoring translation (plane at Z=0).
    Logs cam2pres and K_dst, returns H and output size.
    """
    # Log camera2presentation
    print("07 - camera2presentation matrix:")
    for r in range(4):
        print([f"{cam2pres[r,c]:.5f}" for c in range(4)])

    # Log destination intrinsics
    print("07 - presentation intrinsic K matrix:")
    for r in range(3):
        print([f"{K_dst[r,c]:.5f}" for c in range(3)])

    # Extract rotation only
    R = cam2pres[:3,:3]
    print("07 - Rotation block R (from rig to presentation):")
    for r in range(3):
        print([f"{R[r,c]:.5f}" for c in range(3)])

    # Build homography H = K_dst * R * inv(K_src)
    H = K_dst @ R @ np.linalg.inv(K_src)
    H /= H[2,2]
    print("07 - Computed homography H (rig->presentation):")
    for r in range(3):
        print([f"{H[r,c]:.5f}" for c in range(3)])

    # Determine output size
    # Try using principal point fallback, else rig image size
    cx, cy = K_dst[0,2], K_dst[1,2]
    if cx > 0 and cy > 0:
        out_w = int(2*cx)
        out_h = int(2*cy)
    else:
        out_h, out_w = fallback_size
    print(f"07 - Output presentation size: width={out_w}, height={out_h}")
    return H, (out_w, out_h)


def apply_presentation(rig_img: np.ndarray,
                       data: dict,
                       workdir: Path,
                       idx: int) -> np.ndarray:
    # Load camera2presentation
    vals = list(map(float, data['alignment']['camera2presentation'].split(',')))
    cam2pres = np.array(vals, dtype=float).reshape((4,4))

    # Load or fallback presentation intrinsics
    if 'presentation_intrinsic' in data['alignment']:
        vals2 = list(map(float, data['alignment']['presentation_intrinsic'].split(',')))
        K_dst = np.array(vals2, dtype=float).reshape((3,3))
    else:
        K_dst = np.eye(3)
        print("07 - presentation_intrinsic missing, using identity.")

    # rig image size for fallback
    h_vis, w_vis = rig_img.shape[:2]

    # Compute homography and output size
    H, (out_w, out_h) = compute_homography_presentation(np.eye(3), cam2pres, K_dst, (w_vis, h_vis))

    # Warp rig image
    final = cv2.warpPerspective(rig_img, H, (out_w, out_h))

    # Save
    out_path = workdir / f"frame{idx:06d}_07_presentation.png"
    Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)).save(out_path)
    print(f"07 - Saved presentation view: {out_path}")

    return final

