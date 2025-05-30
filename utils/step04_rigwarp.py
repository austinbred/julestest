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
import camlogger

# Configuration flags
COL_MAJOR = False               # If True, treat extrinsic as column-major
CONVERT_FROM_THREEJS = True     # If True, convert extrinsics from Three.js axes to OpenCV axes
ROTATION_SCALE = 1.0            # Scale applied to extrinsic rotation angles (1.0 = no scaling)
INVERT_EXTRINSIC = False        # If True, invert the extrinsic matrix
REMOVE_ROTATION = False         # If True, ignore rotation (pure translation)
NP_EYE_FOR_DEBUG = False        # If True, use np.eye() for debugging instead of actual matrices


def compute_homography(K_src: np.ndarray,
                       extrinsic: np.ndarray,
                       K_rig: np.ndarray,
                       Z_ref: float) -> np.ndarray:
    """
    Compute a 3x3 homography H mapping the source image to the rig frame,
    accounting for rotation R and translation t with respect to a reference plane at Z_ref.

    H = K_rig * (R - t * n^T / Z_ref) * inv(K_src)
    where n = [0, 0, 1]^T is the plane normal, and [R|t] is the 4x4 extrinsic.
    """

    if INVERT_EXTRINSIC:
        extrinsic = np.linalg.inv(extrinsic)
        print("04 - Extrinsic matrix inverted for lens->rig coordinate mapping.")

    # Extract rotation and translation
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3].reshape(3,1)

    # Optionally scale rotation
    if ROTATION_SCALE != 1.0:
        # Convert to Rodrigues, scale, convert back
        rvec, _ = cv2.Rodrigues(R)
        R, _ = cv2.Rodrigues(rvec * ROTATION_SCALE)
        camlogger.log_extrinsics('ScaledSrcCam', np.vstack([np.hstack([R, t]), [0,0,0,1]]))

    # Reference plane normal and planar homography term
    n = np.array([[0.0, 0.0, 1.0]])
    P_plane = R - (t @ n) / Z_ref

    # Determine effective rig intrinsics
    if np.allclose(K_rig, np.eye(3)):
        K_eff = K_src
    else:
        K_eff = K_rig

    # Compute homography
    H = K_eff @ P_plane @ np.linalg.inv(K_src)
    H /= H[2, 2]

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
    Uses module-level flags for extrinsic interpretation and axis conventions.
    """

    # Log camera parameters
    camlogger.log_intrinsics('SrcCam', K_src)
    camlogger.log_intrinsics('RigCam', K_rig)
    camlogger.log_extrinsics('Original extrinsic', extrinsic)


    # Handle column-major extrinsic if configured
    if COL_MAJOR:
        extrinsic = extrinsic.T.copy()
        print(f"04 - Extrinsic transposed for column-major on {side} lens.")
        camlogger.log_extrinsics("Extrinsic transposed to column-major",extrinsic)

    # Handle Three.js axis conversion if configured
    if CONVERT_FROM_THREEJS:
        S = np.diag([1.0, -1.0, -1.0])
        R_orig = extrinsic[:3, :3]
        t_orig = extrinsic[:3, 3]
        R_conv = S @ R_orig @ S
        t_conv = S @ t_orig
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_conv
        extrinsic[:3, 3] = t_conv
        print(f"04 - Converted extrinsic axes from Three.js to OpenCV on {side} lens.")
        camlogger.log_extrinsics("Converted from Three.js", extrinsic)

    # Load image if a file path is provided
    if isinstance(undistorted, (str, Path)):
        img_pil = Image.open(undistorted)
        src_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        src_bgr = undistorted

    if NP_EYE_FOR_DEBUG:
        # For debugging, use identity matrices
        print("04 - Using np.eye() for extrinsic, K_src, K_rig, Z_ref for debugging.")
        #remove extrinsics rotation
        #extrinsic[0:3,0:3] = 0.0
        #extrinsic[0:3,0:3] = np.eye(3)
        extrinsic = np.eye(4)
        K_src = np.eye(3)
        K_rig = np.eye(3)
        Z_ref = 0.00039
        camlogger.log_extrinsics("np.eye'd extrinsic",extrinsic)
        camlogger.log_intrinsics("np.eye'd K_src",K_src)
        camlogger.log_intrinsics("Znp.eye'd K_rig",K_rig)
   
    # Log extrinsics for this side
    print(f"04 - Z_ref = {Z_ref:0.6f}")

    # Compute homography
    H = compute_homography(K_src, extrinsic, K_rig, Z_ref)
    print(f"04 - Homography to rig for {side} lens applied.")

    # Log expected pixel offset from homography
    offset_x = H[0,2]
    offset_y = H[1,2]
    print(f"04 - Expected pixel offset -> x: {offset_x:0.6f}, y: {offset_y:0.6f}")

    # Warp image
    #h, w = src_bgr.shape[:2]
    #print(f"04 - Warping {side} image of size {w}x{h}.")
    #rig_bgr = cv2.warpPerspective(src_bgr, H, (w, h))

            # Warp onto 2× canvas, aligning top and side per lens
    h, w = src_bgr.shape[:2]
    canvas_w, canvas_h = w * 3, h * 3
    # Compute translation to place image on left or right half (top-aligned)
    if side.lower() == 'left':
        tx, ty = 0, 0
    else:  # right lens
        tx, ty = 2*w, 0
    # Build translation matrix to shift output
    T_center = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
    # Prepend translation so homography maps into larger canvas
    H_big = T_center @ H
    print(f"04 - Warping {side} onto canvas {canvas_w}x{canvas_h} at offset ({tx},{ty}).")
    rig_bgr = cv2.warpPerspective(src_bgr, H_big, (canvas_w, canvas_h))



    # Save rig-space image
    out_path = workdir / f"frame{idx:06d}_04_{side}_rig.png"
    Image.fromarray(cv2.cvtColor(rig_bgr, cv2.COLOR_BGR2RGB)).save(out_path)
    print(f"04 - Saved {side} rig-space image: {out_path}")

    return rig_bgr



