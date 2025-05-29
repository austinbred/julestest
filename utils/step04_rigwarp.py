# step04_rigwarp.py
"""
Module: step04_rigwarp.py

Warps each undistorted lens image into the common rig coordinate frame
using the respective lens2camera extrinsic matrices and rig intrinsics.
Logs all input matrices, computed homographies (including translation for a reference plane),
and saves the rig-space images.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import math


def compute_homography(K_src: np.ndarray,
                       extrinsic: np.ndarray,
                       K_rig: np.ndarray,
                       Z_ref: float) -> np.ndarray:
    """
    Compute a 3x3 homography H mapping the source image to the rig frame,
    accounting for rotation R and translation t with respect to a reference plane at Z_ref.

    H = K_rig * (R - t * n^T / Z_ref) * inv(K_src)
    where n = [0, 0, 1]^T is the plane normal, and [R|t] is the 4x4 extrinsic.
    Logs all matrices and parameters with five-decimal precision.
    """
    # Log source intrinsic matrix K_src
    ksrc_labels = [f"ks{r}{c}={K_src[r,c]:.5f}" for r in range(3) for c in range(3)]
    print("04 - K_src (lens intrinsic):")
    for i in range(3):
        print("[" + ", ".join(ksrc_labels[i*3:(i+1)*3]) + "]")

    # Log rig intrinsic matrix K_rig
    krig_labels = [f"kr{r}{c}={K_rig[r,c]:.5f}" for r in range(3) for c in range(3)]
    print("04 - K_rig (rig intrinsic):")
    for i in range(3):
        print("[" + ", ".join(krig_labels[i*3:(i+1)*3]) + "]")

    # Log extrinsic matrix
    print("04 - Extrinsic (4x4 lens2camera):")
    ext_labels = [f"e{r}{c}={extrinsic[r,c]:.5f}" for r in range(4) for c in range(4)]
    for i in range(4):
        print("[" + ", ".join(ext_labels[i*4:(i+1)*4]) + "]")

    # Extract R and t
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3].reshape(3,1)
    print(f"04 - Translation vector t: [{t[0,0]:.5f}, {t[1,0]:.5f}, {t[2,0]:.5f}]")
    print(f"04 - Reference plane depth Z_ref: {Z_ref:.5f}")

    # Build planar homography term: R - (t * n^T) / Z_ref
    n = np.array([[0.0, 0.0, 1.0]])  # row vector
    P_plane = R - (t @ n) / Z_ref
    # Log plane projection matrix
    pll_labels = [f"p{r}{c}={P_plane[r,c]:.5f}" for r in range(3) for c in range(3)]
    print("04 - Plane projection (R - t*n^T/Z_ref):")
    for i in range(3):
        print("[" + ", ".join(pll_labels[i*3:(i+1)*3]) + "]")

    # Compute homography
    H = K_rig @ P_plane @ np.linalg.inv(K_src)
    H = H / H[2,2]
    # Log computed homography
    h_labels = [f"h{r}{c}={H[r,c]:.5f}" for r in range(3) for c in range(3)]
    print("04 - Computed homography H (with translation):")
    for i in range(3):
        print("[" + ", ".join(h_labels[i*3:(i+1)*3]) + "]")

    return H


def warp_to_rig(undistorted,
                K_src: np.ndarray,
                extrinsic: np.ndarray,
                K_rig: np.ndarray,
                Z_ref: float,
                workdir: Path,
                idx: int,
                side: str) -> np.ndarray:
    """
    Warps one undistorted image (Path or BGR array) into rig frame.
    Uses reference-plane-aware homography and logs all steps.
    """
    # Load image if a file path is provided
    if isinstance(undistorted, (str, Path)):
        img_pil = Image.open(undistorted)
        src_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        src_bgr = undistorted

    # Compute homography including translation
    H = compute_homography(K_src, extrinsic, K_rig, Z_ref)
    print(f"04 - Homography to rig for {side} lens applied.")

    # Warp image to same size as source
    h, w = src_bgr.shape[:2]
    print(f"04 - Warping {side} image of size {w}x{h}.")
    rig_bgr = cv2.warpPerspective(src_bgr, H, (w, h))

    # Save rig-space image
    out_path = workdir / f"frame{idx:06d}_04_{side}_rig.png"
    Image.fromarray(cv2.cvtColor(rig_bgr, cv2.COLOR_BGR2RGB)).save(out_path)
    print(f"04 - Saved {side} rig-space image: {out_path}")

    return rig_bgr

