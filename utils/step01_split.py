# step01_split.py
"""
Module: step01_split.py

Splits a BGR video frame into left and right halves, corrects color, and saves as PNGs.
Also handles scaling of intrinsic matrices based on actual image dimensions.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import camlogger

# Original sensor dimensions from VEO calibration
SENSOR_WIDTH = 3840
SENSOR_HEIGHT = 2160

def scale_intrinsics(K: np.ndarray, actual_width: int, actual_height: int) -> np.ndarray:
    """
    Scales intrinsic matrix from sensor resolution to actual image resolution.
    
    Args:
        K: Original intrinsic matrix from sensor resolution (3840×2160)
        actual_width: Width of the actual image
        actual_height: Height of the actual image
        
    Returns:
        K_scaled: Scaled intrinsic matrix
    """
    # Log original dimensions
    camlogger.logger.info("")
    camlogger.logger.info("=== Intrinsic Matrix Scaling ===")
    camlogger.logger.info(f"Original sensor dimensions: {SENSOR_WIDTH}×{SENSOR_HEIGHT}")
    camlogger.logger.info(f"Actual image dimensions: {actual_width}×{actual_height}")
    
    # Calculate scaling factors
    scale_x = actual_width / SENSOR_WIDTH
    scale_y = actual_height / SENSOR_HEIGHT
    
    camlogger.logger.info(f"Scaling factors: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
    
    # Scale focal lengths and principal point
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    
    return K_scaled

def split_frame(frame: np.ndarray, Kl: np.ndarray, Kr: np.ndarray, workdir: Path, idx: int):
    """
    Splits a BGR frame into left and right halves, converts to RGB,
    saves with a 01 step filename, and optionally scales intrinsic matrices.

    Args:
        frame (np.ndarray): BGR image frame from OpenCV.
        Kl (np.ndarray): Left camera intrinsic matrix to scale
        Kr (np.ndarray): Right camera intrinsic matrix to scale
        workdir (Path): Directory to save outputs.
        idx (int): Frame index for output filenames.

    Returns:
        tuple: (left_img, right_img, Kl_scaled, Kr_scaled)
            - left_img: Left half of frame as numpy array
            - right_img: Right half of frame as numpy array
            - Kl_scaled: Scaled left intrinsic matrix
            - Kr_scaled: Scaled right intrinsic matrix
    """
    camlogger.logger.info("")
    camlogger.logger.info("=== step01_split.py: Starting frame split and intrinsic scaling ===")
    
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    camlogger.logger.info(f"Input frame size: width={w}, height={h}")
    
    # Split top (left camera) and bottom (right camera)
    left = frame_rgb[:h//2, :, :]
    right = frame_rgb[h//2:, :, :]
    camlogger.logger.info(f"Split into halves: width={w}, height={h//2} each")
    
    # Save outputs
    out_l = workdir / "01_left.png"
    out_r = workdir / "01_right.png"
    Image.fromarray(left).save(out_l)
    Image.fromarray(right).save(out_r)
    camlogger.logger.info(f"Saved split frames: {out_l}, {out_r}")
    
    # Scale intrinsic matrices
    camlogger.logger.info("")
    camlogger.logger.info("Scaling left camera intrinsics:")
    camlogger.log_intrinsics(Kl, "Original left (sensor resolution)")
    Kl_scaled = scale_intrinsics(Kl, w, h//2)
    camlogger.log_intrinsics(Kl_scaled, "Scaled left (actual resolution)")
    
    camlogger.logger.info("")
    camlogger.logger.info("Scaling right camera intrinsics:")
    camlogger.log_intrinsics(Kr, "Original right (sensor resolution)")
    Kr_scaled = scale_intrinsics(Kr, w, h//2)
    camlogger.log_intrinsics(Kr_scaled, "Scaled right (actual resolution)")
    
    camlogger.logger.info("")
    camlogger.logger.info("=== step01_split.py completed successfully ===")
    
    return left, right, Kl_scaled, Kr_scaled

