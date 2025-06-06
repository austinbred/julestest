"""
Module: camlogger.py

Provides logging utilities for camera calibration pipeline.
Logs both to console and a single file, with different formats for each.
"""
import numpy as np
import cv2
import logging
from pathlib import Path
import os
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Use a single log file
log_file = log_dir / 'camera_calibration.log'

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters
console_formatter = logging.Formatter('%(message)s')  # Simple format for console
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Detailed format for file

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# File handler - use append mode ('a')
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log a session start marker
logger.info("\n" + "="*80)
logger.info(f"Starting new calibration session at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80 + "\n")

def log_distortion(name: str, D: np.ndarray, is_opencv: bool = False) -> None:
    """
    Logs the distortion matrix with format depending on calibration type:
    OpenCV (5 coefficients): [k1, k2, k3, p1, p2]
    Fisheye (4 coefficients): [k1, k2, k3, k4]
    """
    D = np.array(D, dtype=float).flatten()
    
    logger.info("")
    logger.info(f"{name} Distortion ({['Fisheye', 'OpenCV'][is_opencv]} format):")
    if is_opencv:
        # OpenCV format: k1, k2, k3, p1, p2
        if len(D) >= 5:
            logger.info(f"[ k1={D[0]:12.6f}, k2={D[1]:12.6f}, k3={D[2]:12.6f}, p1={D[3]:12.6f}, p2={D[4]:12.6f} ]")
        else:
            logger.info(f"Warning: Expected 5 coefficients for OpenCV format, got {len(D)}")
            logger.info(f"Values: {D}")
    else:
        # Fisheye format: k1, k2, k3, k4
        if len(D) >= 4:
            logger.info(f"[ k1={D[0]:12.6f}, k2={D[1]:12.6f}, k3={D[2]:12.6f}, k4={D[3]:12.6f} ]")
        else:
            logger.info(f"Warning: Expected 4 coefficients for fisheye format, got {len(D)}")
            logger.info(f"Values: {D}")


def log_intrinsics(K: np.ndarray, name: str) -> None:
    """
    Logs the intrinsic matrix with standard camera calibration parameter names:
      [ fx=<k00>,   s=<k01>,  cx=<k02> ]
      [  0=<k10>,  fy=<k11>,  cy=<k12> ]
      [  0=<k20>,   0=<k21>,   1=<k22> ]
    
    Where:
    - fx, fy: Focal lengths in x and y
    - cx, cy: Principal point coordinates
    - s: Skew (usually 0)
    """
    K = np.array(K, dtype=float)
    if K.shape != (3, 3):
        logger.warning(f"Warning: Expected 3x3 matrix, got {K.shape}")
        return

    logger.info("")
    logger.info(f"{name} Intrinsic Matrix:")
    
    # Row 1: [fx  s  cx]
    logger.info(f"[ fx={K[0,0]:12.6f},  s={K[0,1]:12.6f}, cx={K[0,2]:12.6f} ]")
    
    # Row 2: [ 0 fy  cy]
    logger.info(f"[  0={K[1,0]:12.6f}, fy={K[1,1]:12.6f}, cy={K[1,2]:12.6f} ]")
    
    # Row 3: [ 0  0   1]
    logger.info(f"[  0={K[2,0]:12.6f},  0={K[2,1]:12.6f},  1={K[2,2]:12.6f} ]")


def log_extrinsics(name: str, E: np.ndarray) -> None:
    """
    Logs the extrinsic matrix with :
      [ e00=<e00>, e01=<e01>, e02=<e02>, e03=<e03> ]
      [ e10=<e10>, e11=<e11>, e12=<e12>, e13=<e13> ]
      [ e20=<e20>, e21=<e21>, e22=<e22>, e23=<e23> ]
      [ e30=<e30>, e31=<e31>, e32=<e32>, e33=<e33> ]
    """
    E = np.array(E, dtype=float)
    if E.shape != (4, 4):
        logger.warning(f"Warning: Expected 4x4 matrix, got {E.shape}")
        return

    # Log rotation rows
    logger.info("")
    logger.info(f"{name} Extrinsic Matrix:")
    for i in range(4):
        row = E[i]
        labels = [f"e{i}{j}={row[j]:12.6f}" for j in range(4)]
        logger.info(f"[ {', '.join(labels)} ]")

