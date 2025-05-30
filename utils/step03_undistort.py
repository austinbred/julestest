import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple

def undistort_fisheye(input_path: Path,
                      K: np.ndarray,
                      D: np.ndarray,
                      workdir: Path,
                      idx: int,
                      side: str) -> Path:
    """
    Undistort a fisheye image and save:
      - Full undistorted image (with black borders)
      - Padded-region mask (central 80% of image)
      - Masked undistorted image

    Returns:
        full_path (Path): Path to the full undistorted image
        mask_path (Path): Path to the saved mask image
        masked_path (Path): Path to the masked undistorted image
    """
    # Load image and convert to BGR
    img_rgb = Image.open(input_path)
    img = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # Undistort using fisheye model
    und_full = cv2.fisheye.undistortImage(img, K, D, Knew=K)

    # Save full undistorted image
    full_name = f"frame{idx:06d}_03_{side}_ud_full.png"
    full_path = workdir / full_name
    cv2.imwrite(str(full_path), und_full)
    print(f"03 - Saved full undistorted image: {full_path}")

    # Create mask: full black, white in central 80% region (remove 10% border)
    pad_h = int(0.22 * h)
    pad_w = int(0.20 * w)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[pad_h:h-pad_h, pad_w:w-pad_w] = 255

    # Save mask
    mask_name = f"frame{idx:06d}_03_{side}_mask.png"
    mask_path = workdir / mask_name
    cv2.imwrite(str(mask_path), mask)
    print(f"03 - Saved padded-region mask: {mask_path}")

    # Apply mask to undistorted image
    und_masked = cv2.bitwise_and(und_full, und_full, mask=mask)
    masked_name = f"frame{idx:06d}_03_{side}_ud_masked.png"
    masked_path = workdir / masked_name
    cv2.imwrite(str(masked_path), und_masked)
    print(f"03 - Saved masked undistorted image: {masked_path}")

    return masked_path

