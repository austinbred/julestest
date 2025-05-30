import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from camlogger import log_intrinsics, log_distortion

def undistort_fisheye(input_path: Path,
                      K: np.ndarray,
                      D: np.ndarray,
                      workdir: Path,
                      idx: int,
                      side: str) -> Path:
    """
    Undistort a fisheye image using an optimal new camera matrix,
    then apply the padded-region mask, and save:
      - Full undistorted image
      - Padded-region mask
      - Masked undistorted image

    Returns:
        masked_path (Path)
    """
    # Load image and convert to BGR
    img = cv2.cvtColor(np.array(Image.open(input_path)), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # Log original intrinsics & distortion
    log_intrinsics(f"K_{side}", K)
    K[0, 2] = w / 2  # center x
    K[1, 2] = h / 2  # center y
    log_intrinsics(f"Modified K_{side}", K)
    D = np.array(D, dtype=np.float64).flatten()[:4].reshape(4, 1)
    # Initial values for D: [ -0.800357, -0.109341, -0.001276, -1.155151]    
    D[0]=D[0]*0.7
    D[1]=D[1]*0.7
    D[2]=D[2]*0.7
    D[3]=D[3]*0.7
    log_distortion(f"D_{side}", D)

    # Re-center the principal point in a new camera matrix
    Knew = K.copy()
    #Knew[0, 2] = w / 2  # center x
    #Knew[1, 2] = h / 2  # center y
    log_intrinsics(f"Knew_{side}", Knew)

    # Undistort image with re-centered Knew
    und_full = cv2.fisheye.undistortImage(img, K, D, Knew=Knew)
    full_path = workdir / f"frame{idx:06d}_03_{side}_ud_full.png"
    cv2.imwrite(str(full_path), und_full)
    print(f"03 - Saved full undistorted image: {full_path}")

    # Apply original padded-region mask (central 80% per tuning)
    pad_h = int(0.22 * h)
    pad_w = int(0.20 * w)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[pad_h:h-pad_h, pad_w:w-pad_w] = 255
    mask_path = workdir / f"frame{idx:06d}_03_{side}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"03 - Saved padded-region mask: {mask_path}")

    # Mask the undistorted image
    und_masked = cv2.bitwise_and(und_full, und_full, mask=mask)
    masked_path = workdir / f"frame{idx:06d}_03_{side}_ud_masked.png"
    cv2.imwrite(str(masked_path), und_masked)
    print(f"03 - Saved masked undistorted image: {masked_path}")

    return masked_path

