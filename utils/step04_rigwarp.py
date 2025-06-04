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


def flip_extrinsic_signs(extrinsic: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the in-plane roll (rotation around Z axis) and yaw (rotation around Y axis)
    in the extrinsic matrix.
    """
    # Extract current 3x3 rotation block
    R = extrinsic[:3, :3].copy()
    # Decompose R into Euler angles via RQ decomposition (angles in degrees)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    rx, ry, rz = angles
    # Flip the sign of the roll (Z-axis) and yaw (Y-axis)
    rz = -rz
    ry = -ry
    rx = 0
    print(f"rx= {rx}, ry= {ry}, rz= {rz}")
   
    print("04 - Flipping extrinsic signs")

    # Helper functions to build rotation matrices
    def rot_x(theta_deg):
        t = np.deg2rad(theta_deg)
        return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

    def rot_y(theta_deg):
        t = np.deg2rad(theta_deg)
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    def rot_z(theta_deg):
        t = np.deg2rad(theta_deg)
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    # Reconstruct new rotation in Z-Y-X order
    R_new = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    extrinsic[:3, :3] = R_new
    camlogger.log_extrinsics("Rolled-and-yawed-flipped extrinsic", extrinsic)
    return extrinsic


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

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3].reshape(3, 1)

    if ROTATION_SCALE != 1.0:
        rvec, _ = cv2.Rodrigues(R)
        R, _ = cv2.Rodrigues(rvec * ROTATION_SCALE)
        camlogger.log_extrinsics('ScaledSrcCam', np.vstack([np.hstack([R, t]), [0,0,0,1]]))

    n = np.array([[0.0, 0.0, 1.0]])
    P_plane = R - (t @ n) / Z_ref

    if np.allclose(K_rig, np.eye(3)):
        K_eff = K_src
    else:
        K_eff = K_rig

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

    camlogger.log_intrinsics('SrcCam', K_src)
    camlogger.log_intrinsics('RigCam', K_rig)
    camlogger.log_extrinsics('Original extrinsic', extrinsic)

    if COL_MAJOR:
        extrinsic = extrinsic.T.copy()
        print(f"04 - Extrinsic transposed for column-major on {side} lens.")
        camlogger.log_extrinsics("Extrinsic transposed to column-major", extrinsic)

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

        # Flip the sign of the in-plane roll for the right lens only

    extrinsic = flip_extrinsic_signs(extrinsic)

    if isinstance(undistorted, (str, Path)):
        img_pil = Image.open(undistorted)
        src_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        src_bgr = undistorted

    camlogger.log_extrinsics("Post-conversion extrinsic", extrinsic)
    camlogger.log_intrinsics("Post-conversion K_src", K_src)
    camlogger.log_intrinsics("Post-conversion K_rig", K_rig)

    print(f"04 - Z_ref = {Z_ref:0.6f}")
    H = compute_homography(K_src, extrinsic, K_rig, Z_ref)
    print(f"04 - Homography to rig for {side} lens applied.")

    offset_x = H[0, 2]
    offset_y = H[1, 2]
    print(f"04 - Expected pixel offset -> x: {offset_x:0.6f}, y: {offset_y:0.6f}")

    h, w = src_bgr.shape[:2]
    canvas_w, canvas_h = w * 2, h * 2
    if side.lower() == 'left':
        tx, ty = 1.5*w, -0.33*h
    else:
        tx, ty = -0.33*w, -0.33*h
    T_center = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
    H_big = T_center @ H
    print(f"04 - Warping {side} onto canvas {canvas_w}x{canvas_h} at offset ({tx},{ty}).")
    rig_bgr = cv2.warpPerspective(src_bgr, H_big, (canvas_w, canvas_h))

    out_path = workdir / f"frame{idx:06d}_04_{side}_rig.png"
    Image.fromarray(cv2.cvtColor(rig_bgr, cv2.COLOR_BGR2RGB)).save(out_path)
    print(f"04 - Saved {side} rig-space image: {out_path}")

    return rig_bgr
