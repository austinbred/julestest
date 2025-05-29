# step04_rectify.py
"""
Module: step04_rectify.py

Performs stereo rectification on undistorted images, handling color conversion and
logging all inputs, intermediate matrices, and outputs with two-decimal labels.
"""
import cv2
import numpy as np
import math
from pathlib import Path
from PIL import Image


def rectify_stereo(left_path: Path, right_path: Path,
                   Kl: np.ndarray, Kr: np.ndarray,
                   R_rel: np.ndarray, T_px: float,
                   workdir: Path, idx: int):
    """
    Loads left/right undistorted PNGs, converts to BGR for OpenCV,
    applies stereo rectification with separate intrinsics and detailed logging,
    converts back to RGB and saves rectified outputs.

    Args:
        left_path (Path): Path to left undistorted image.
        right_path (Path): Path to right undistorted image.
        Kl (np.ndarray): 3×3 intrinsic matrix for left camera.
        Kr (np.ndarray): 3×3 intrinsic matrix for right camera.
        R_rel (np.ndarray): Relative rotation matrix (left→right).
        T_px (float): Baseline shift in pixels along X.
        workdir (Path): Directory for saving outputs.
        idx (int): Frame index for naming.

    Returns:
        tuple: (R0_bgr, R1_bgr) rectified BGR arrays.
    """
    # Load images via PIL (RGB) and convert to BGR for OpenCV
    left_pil = Image.open(left_path)
    right_pil = Image.open(right_path)
    arr_l = cv2.cvtColor(np.array(left_pil), cv2.COLOR_RGB2BGR)
    arr_r = cv2.cvtColor(np.array(right_pil), cv2.COLOR_RGB2BGR)

    # Log intrinsics Kl and Kr
    def log_matrix(name, M):
        labels = [f"{name}{r}{c}={M[r,c]:.2f}" for r in range(M.shape[0]) for c in range(M.shape[1])]
        print(f"04 - {name} matrix:")
        for i in range(M.shape[0]):
            row = labels[i*M.shape[1]:(i+1)*M.shape[1]]
            print("[" + ", ".join(row) + "]")
    log_matrix('Kl', Kl)
    log_matrix('Kr', Kr)

    # Log relative rotation before any override
    r_js = R_rel.copy()
    rjs_labels = [f"rjs{r}{c}={r_js[r,c]:.2f}" for r in range(3) for c in range(3)]
    print("04 - R_js (initial relative rotation):")
    for i in range(3):
        row = rjs_labels[i*3:(i+1)*3]
        print("[" + ", ".join(row) + "]")

    # Decompose R_js into yaw, pitch, roll (ZYX)
    if abs(r_js[2,0]) < 1.0:
        pitch = math.asin(-r_js[2,0])
        yaw   = math.atan2(r_js[1,0], r_js[0,0])
        roll  = math.atan2(r_js[2,1], r_js[2,2])
    else:
        pitch = math.pi/2 * (-1 if r_js[2,0] > 0 else 1)
        yaw   = math.atan2(-r_js[0,1], r_js[1,1])
        roll  = 0.0
    print(f"04 - Decomposed R_js -> yaw={math.degrees(yaw):.2f}°, pitch={math.degrees(pitch):.2f}°, roll={math.degrees(roll):.2f}°")

        # Override R_rel to a 90.5° rotation about Y-axis for testing
    theta = math.radians(90.84)
    cy = math.cos(theta)
    sy = math.sin(theta)
    # Rotation about Y-axis
    R_override = np.array([
        [ cy, 0.0,  sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0,  cy]
    ], dtype=float)
#    R_rel = R_override
    print("04 - Overriding R_rel to a 90.5° rotation about Y-axis for testing.")

    # Log final R_rel
    rrel_labels = [f"rrel{r}{c}={R_rel[r,c]:.2f}" for r in range(3) for c in range(3)]
    print("04 - R_rel (used in stereoRectify):")
    for i in range(3):
        row = rrel_labels[i*3:(i+1)*3]
        print("[" + ", ".join(row) + "]")

    # Log baseline translation vector
    print(f"04 - Baseline T_px: [{T_px:.2f}]")

    # Zero distortion vectors
    zero_l = np.zeros(5); zero_r = np.zeros(5)
    print(f"04 - Zero distortion: left={list(zero_l)}, right={list(zero_r)}")

    # Perform stereo rectification
    size = (arr_l.shape[1], arr_l.shape[0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, zero_l,
        Kr, zero_r,
        size,
        R_rel, np.array([T_px, 0.0, 0.0]),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.5
    )

    # Log projection matrices P1 and P2
    def log_proj(name, P):
        labels = [f"{name}{r}{c}={P[r,c]:.2f}" for r in range(P.shape[0]) for c in range(P.shape[1])]
        print(f"04 - {name} matrix:")
        for i in range(P.shape[0]):
            row = labels[i*P.shape[1]:(i+1)*P.shape[1]]
            print("[" + ", ".join(row) + "]")
    log_proj('P1', P1)
    log_proj('P2', P2)

    # Log ROIs
    print(f"04 - ROI1={roi1}, ROI2={roi2}")
    if roi1[2] == 0 or roi1[3] == 0:
        print("Warning: ROI1 is empty, rectified left may be blank.")
    if roi2[2] == 0 or roi2[3] == 0:
        print("Warning: ROI2 is empty, rectified right may be blank.")

    # Generate undistort-rectify maps and apply
    map1l, map2l = cv2.initUndistortRectifyMap(Kl, zero_l, R1, P1, size, cv2.CV_16SC2)
    map1r, map2r = cv2.initUndistortRectifyMap(Kr, zero_r, R2, P2, size, cv2.CV_16SC2)
    R0_bgr = cv2.remap(arr_l, map1l, map2l, cv2.INTER_LINEAR)
    R1_bgr = cv2.remap(arr_r, map1r, map2r, cv2.INTER_LINEAR)

    # Save rectified outputs via PIL
    out_l = workdir / f"frame{idx:06d}_04_left_rectified.png"
    out_r = workdir / f"frame{idx:06d}_04_right_rectified.png"
    Image.fromarray(cv2.cvtColor(R0_bgr, cv2.COLOR_BGR2RGB)).save(out_l)
    Image.fromarray(cv2.cvtColor(R1_bgr, cv2.COLOR_BGR2RGB)).save(out_r)
    print(f"04 - Saved rectified frames: {out_l}, {out_r}")

    return R0_bgr, R1_bgr

