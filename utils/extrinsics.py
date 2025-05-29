# extrinsics.py
"""
Module: extrinsics.py

Computes the relative rotation and translation between left and right cameras,
with detailed logging of all intermediate matrices.
"""
import numpy as np


def parse_extrinsics(align: dict, cam2pres: np.ndarray = None):
    """
    Calculate R_rel, T_rel in OpenCV coords, ignoring cam2pres.
    Logs M_left, M_right, M_rel, R_js, and R_rel with labeled entries.

    Args:
        align (dict): alignment data from calibration JSON.
        cam2pres (np.ndarray): (ignored) presentation-space transform.

    Returns:
        R_rel (np.ndarray): 3×3 relative rotation (left→right).
        T_rel (np.ndarray): 3-vector relative translation.
    """
    def parse_M(key: str) -> np.ndarray:
        vals = list(map(float, align[key].split(',')))
                # JSON values are row-major; reshape accordingly
        M = np.array(vals, dtype=float).reshape((4, 4))
        print(f"Extrinsics: Parsed {key} matrix (row-major reshape):")
        labels = [f"m{r}{c}={M[r,c]:.2f}" for r in range(4) for c in range(4)]
        for i in range(4):
            row = labels[i*4:(i+1)*4]
            print("[" + ", ".join(row) + "]")
        return M

    # Parse lens-to-camera matrices
    M_left  = parse_M('lens_left2camera')
    M_right = parse_M('lens_right2camera')

        # If a presentation-space matrix is provided, apply it before computing relative transform
    if cam2pres is not None:
        print("Extrinsics: Applying camera2presentation matrix:")
        pres_labels = [f"cp{r}{c}={cam2pres[r,c]:.2f}" for r in range(4) for c in range(4)]
        for i in range(4):
            row = pres_labels[i*4:(i+1)*4]
            print("[" + ", ".join(row) + "]")
        M1p = cam2pres @ M_left
        M2p = cam2pres @ M_right
        print("Extrinsics: M1p (cam2pres @ M_left):")
        labels1 = [f"m1p{r}{c}={M1p[r,c]:.2f}" for r in range(4) for c in range(4)]
        for i in range(4): print("[" + ", ".join(labels1[i*4:(i+1)*4]) + "]")
        print("Extrinsics: M2p (cam2pres @ M_right):")
        labels2 = [f"m2p{r}{c}={M2p[r,c]:.2f}" for r in range(4) for c in range(4)]
        for i in range(4): print("[" + ", ".join(labels2[i*4:(i+1)*4]) + "]")
        # compute relative in presentation space
        M_rel = np.linalg.inv(M1p) @ M2p
    else:
        # No camera2presentation: compute direct relative transform
        M_rel = np.linalg.inv(M_left) @ M_right
    print("Extrinsics: M_rel (4×4 left→right):")
    labels = [f"mrel{r}{c}={M_rel[r,c]:.2f}" for r in range(4) for c in range(4)]
    for i in range(4):
        row = labels[i*4:(i+1)*4]
        print("[" + ", ".join(row) + "]")

    # Extract rotation from M_rel
    R_js = M_rel[:3, :3]
    print("Extrinsics: R_js (3×3 rotation from M_rel):")
    rjs_labels = [f"rjs{r}{c}={R_js[r,c]:.2f}" for r in range(3) for c in range(3)]
    for i in range(3):
        row = rjs_labels[i*3:(i+1)*3]
        print("[" + ", ".join(row) + "]")

    # Decompose R_js into yaw, pitch, roll (ZYX convention)
    import math
    r = R_js
    if abs(r[2,0]) < 1.0:
        pitch = math.asin(-r[2,0])
        yaw   = math.atan2(r[1,0], r[0,0])
        roll  = math.atan2(r[2,1], r[2,2])
    else:
        # Gimbal lock: pitch = ±90°
        pitch = math.pi/2 * (-1 if r[2,0] > 0 else 1)
        yaw   = math.atan2(-r[0,1], r[1,1])
        roll  = 0.0
    # Convert to degrees
    yaw_deg   = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg  = math.degrees(roll)
    print(f"Extrinsics: R_js Euler ZYX -> yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°, roll={roll_deg:.2f}°")

        # Convert from Three.js → OpenCV coords
    flip = np.diag([1, -1, -1])
    R_rel = flip @ R_js @ flip
    print("Extrinsics: R_rel (Three.js→OpenCV):")
    rrel_labels = [f"rrel{r}{c}={R_rel[r,c]:.2f}" for r in range(3) for c in range(3)]
    for i in range(3):
        row = rrel_labels[i*3:(i+1)*3]
        print("[" + ", ".join(row) + "]")

    # Decompose R_rel into yaw, pitch, roll (ZYX convention)
    import math
    r2 = R_rel
    if abs(r2[2,0]) < 1.0:
        pitch2 = math.asin(-r2[2,0])
        yaw2   = math.atan2(r2[1,0], r2[0,0])
        roll2  = math.atan2(r2[2,1], r2[2,2])
    else:
        # Gimbal lock: pitch = ±90°
        pitch2 = math.pi/2 * (-1 if r2[2,0] > 0 else 1)
        yaw2   = math.atan2(-r2[0,1], r2[1,1])
        roll2  = 0.0
    # Convert to degrees
    yaw2_deg   = math.degrees(yaw2)
    pitch2_deg = math.degrees(pitch2)
    roll2_deg  = math.degrees(roll2)
    print(f"Extrinsics: R_rel Euler ZYX -> yaw={yaw2_deg:.2f}°, pitch={pitch2_deg:.2f}°, roll={roll2_deg:.2f}°")

    # Log translation vectors
    T_js = M_rel[:3, 3]
    print(f"Extrinsics: T_js (before flip) = [{T_js[0]:.2f}, {T_js[1]:.2f}, {T_js[2]:.2f}]")
    T_rel = flip @ T_js
    print(f"Extrinsics: T_rel (after flip)  = [{T_rel[0]:.2f}, {T_rel[1]:.2f}, {T_rel[2]:.2f}]")

    return R_rel, T_rel

