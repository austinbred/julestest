# load_calibration.py
"""
Module: load_calibration.py

Handles loading of calibration JSON, parsing intrinsics, distortion,
and optional presentation-space extrinsics, with detailed logging.
Supports both OpenCV and Three.js calibration formats.
"""
import json
import numpy as np
from pathlib import Path
import camlogger
from camlogger import log_intrinsics, log_distortion, log_extrinsics

def log_raw_calibration(data: dict):
    """
    Log all raw calibration data from JSON before any adjustments.
    
    Args:
        data (dict): Raw calibration data from JSON
    """
    align = data['alignment']
    calibration_method = align.get('calibration_method', 'three_js').lower()
    is_opencv = calibration_method == 'opencv'
    
    # Log field dimensions and reference plane depth
    Z_ref = align.get('field_length', 105.0)
    field_width = align.get('field_width', 68.0)
    camlogger.logger.info("")
    camlogger.logger.info("=== Field Dimensions ===")
    camlogger.logger.info(f"Field length (Z_ref) = {Z_ref:.6f} meters")
    camlogger.logger.info(f"Field width = {field_width:.6f} meters")
    
    # Log raw intrinsic matrices
    for side in ['left', 'right']:
        K_key = f'intrinsic_{side}'
        K = np.array(list(map(float, align[K_key].split(','))), dtype=float).reshape(3, 3)
        log_intrinsics(K, f"Raw {K_key}")
    
    # Log raw distortion coefficients
    for side in ['left', 'right']:
        D_key = f'distortion_{side}'
        D_raw = np.array(list(map(float, align[D_key].split(','))), dtype=float)
        if is_opencv:
            log_distortion(f"Raw {D_key}", D_raw, is_opencv=True)
        else:
            # For Three.js format, show all 14 coefficients
            coeffs = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
            coeffs.extend([f"unused{i}" for i in range(6)])  # Last 6 elements are unused
            coeff_str = ", ".join([f"{name}={val:10.6f}" for name, val in zip(coeffs, D_raw)])
            camlogger.logger.info(f"Raw {D_key} (Three.js format, 14 elements):")
            camlogger.logger.info(f"[ {coeff_str} ]")
            camlogger.logger.info("")
    
    # Log raw lens-to-camera extrinsic matrices
    for side in ['left', 'right']:
        M_key = f'lens_{side}2camera'
        if M_key in align:
            M = np.array(list(map(float, align[M_key].split(','))), dtype=float).reshape(4, 4)
            log_extrinsics(f"Raw {M_key}", M)
    
    # Log raw camera2presentation matrix if available
    if 'camera2presentation' in align:
        extrinsic = np.array(list(map(float, align['camera2presentation'].split(','))), dtype=float).reshape(4, 4)
        log_extrinsics("Raw camera2presentation", extrinsic)

def adjust_calibration(data: dict, side: str) -> tuple:
    """
    Loads and adjusts calibration data from JSON.
    Handles all necessary transformations in one place:
    - Distortion coefficient conversion and negation
    - Extrinsic matrix axis conversion
    
    Note: 
    - Intrinsic matrix scaling is now handled in step01_split.py
    - Canvas expansion and principal point adjustments are handled in step02_expand.py
    
    Args:
        data (dict): Raw calibration data from JSON
        side (str): Camera side ('left' or 'right')
    
    Returns:
        tuple: (K, D_adjusted, extrinsic_adjusted, lens2cam_adjusted)
            - K: Original intrinsic matrix
            - D_adjusted: Converted distortion coefficients
            - extrinsic_adjusted: Converted extrinsic matrix (if available)
            - lens2cam_adjusted: Converted lens-to-camera matrix (if available)
    """
    align = data['alignment']
    calibration_method = align.get('calibration_method', 'three_js').lower()
    is_opencv = calibration_method == 'opencv'
    
    # 1. Load intrinsic matrix (no adjustment needed)
    K_key = f'intrinsic_{side}'
    K = np.array(list(map(float, align[K_key].split(','))), dtype=float).reshape(3, 3)
    
    # 2. Convert distortion coefficients
    D_key = f'distortion_{side}'
    D_raw = list(map(float, align[D_key].split(',')))
    
    if is_opencv:
        # OpenCV format: Use all 5 coefficients (k1,k2,p1,p2,k3)
        D_adjusted = np.array(D_raw[:5], dtype=float)
    else:
        # Three.js format: [k1, k2, p1, p2, k3, k4, k5, k6, 0, 0, 0, 0, 0, 0]
        # Convert to OpenCV fisheye format: [k1, k2, k3, k4]
        # Note: Three.js uses opposite sign convention
        k1, k2 = -D_raw[0], -D_raw[1]  # First two values
        k3, k4 = -D_raw[6], -D_raw[7]  # k5, k6 in Three.js become k3, k4
        D_adjusted = np.array([k1, k2, k3, k4], dtype=float)
    
    # 3. Convert extrinsic matrix if available
    extrinsic_adjusted = None
    if 'camera2presentation' in align:
        extrinsic = np.array(list(map(float, align['camera2presentation'].split(','))), dtype=float).reshape(4, 4)
        
        # Convert from Three.js to OpenCV coordinate system
        if not is_opencv:
            # Three.js uses Y-up, Z-forward, while OpenCV uses Y-down, Z-forward
            S = np.diag([1.0, -1.0, -1.0])  # Sign change matrix
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            R_adjusted = S @ R @ S
            t_adjusted = S @ t
            
            extrinsic_adjusted = np.eye(4)
            extrinsic_adjusted[:3, :3] = R_adjusted
            extrinsic_adjusted[:3, 3] = t_adjusted
        else:
            extrinsic_adjusted = extrinsic
    
    # 4. Convert lens-to-camera matrix if available
    lens2cam_adjusted = None
    M_key = f'lens_{side}2camera'
    if M_key in align:
        lens2cam = np.array(list(map(float, align[M_key].split(','))), dtype=float).reshape(4, 4)
        
        # Convert from Three.js to OpenCV coordinate system
        if not is_opencv:
            # Three.js uses Y-up, Z-forward, while OpenCV uses Y-down, Z-forward
            S = np.diag([1.0, -1.0, -1.0])  # Sign change matrix
            R = lens2cam[:3, :3]
            t = lens2cam[:3, 3]
            R_adjusted = S @ R @ S
            t_adjusted = S @ t
            
            lens2cam_adjusted = np.eye(4)
            lens2cam_adjusted[:3, :3] = R_adjusted
            lens2cam_adjusted[:3, 3] = t_adjusted
        else:
            lens2cam_adjusted = lens2cam
    
    return K, D_adjusted, extrinsic_adjusted, lens2cam_adjusted

def log_adjusted_calibration(Kl: np.ndarray, Dl: np.ndarray, Kr: np.ndarray, Dr: np.ndarray, 
                           cam2pres: np.ndarray, lens_left2cam: np.ndarray, lens_right2cam: np.ndarray, 
                           is_opencv: bool):
    """
    Log all adjusted calibration data after conversions.
    
    Args:
        Kl: Left camera intrinsic matrix
        Dl: Adjusted left camera distortion coefficients
        Kr: Right camera intrinsic matrix
        Dr: Adjusted right camera distortion coefficients
        cam2pres: Adjusted camera2presentation matrix
        lens_left2cam: Adjusted left lens-to-camera matrix
        lens_right2cam: Adjusted right lens-to-camera matrix
        is_opencv: Whether using OpenCV format
    """
    # Log adjusted intrinsic matrices (no adjustment needed for OpenCV format)
    if is_opencv:
        log_intrinsics(Kl, "Final intrinsic_left (no adjustment needed)")
        log_intrinsics(Kr, "Final intrinsic_right (no adjustment needed)")
    else:
        log_intrinsics(Kl, "Adjusted intrinsic_left")
        log_intrinsics(Kr, "Adjusted intrinsic_right")
    
    # Log adjusted distortion coefficients
    if is_opencv:
        log_distortion("Final distortion_left (no adjustment needed)", Dl, is_opencv)
        log_distortion("Final distortion_right (no adjustment needed)", Dr, is_opencv)
    else:
        log_distortion("Adjusted distortion_left", Dl, is_opencv)
        log_distortion("Adjusted distortion_right", Dr, is_opencv)
    
    # Log adjusted lens-to-camera matrices if available
    if lens_left2cam is not None:
        if is_opencv:
            log_extrinsics("Final lens_left2camera (no adjustment needed)", lens_left2cam)
        else:
            log_extrinsics("Adjusted lens_left2camera", lens_left2cam)
    
    if lens_right2cam is not None:
        if is_opencv:
            log_extrinsics("Final lens_right2camera (no adjustment needed)", lens_right2cam)
        else:
            log_extrinsics("Adjusted lens_right2camera", lens_right2cam)
    
    # Log adjusted extrinsic matrix if available
    if cam2pres is not None:
        if is_opencv:
            log_extrinsics("Final camera2presentation (no adjustment needed)", cam2pres)
        else:
            log_extrinsics("Adjusted camera2presentation", cam2pres)

def load_calibration(path: Path):
    """
    Load and parse calibration data, applying all necessary adjustments.
    
    Args:
        path (Path): Path to calibration JSON file.
    
    Returns:
        data (dict): Full JSON data.
        Kl (np.ndarray): Original left camera intrinsic matrix.
        Dl (np.ndarray): Adjusted left camera distortion coefficients.
        Kr (np.ndarray): Original right camera intrinsic matrix.
        Dr (np.ndarray): Adjusted right camera distortion coefficients.
        cam2pres (np.ndarray or None): Adjusted presentation-space transform.
        lens_left2cam (np.ndarray or None): Adjusted left lens-to-camera transform.
        lens_right2cam (np.ndarray or None): Adjusted right lens-to-camera transform.
        is_opencv (bool): True if using OpenCV calibration format.
        Z_ref (float): Reference plane depth (field length).
    """
    camlogger.logger.info("")
    camlogger.logger.info("=== load_calibration.py: Loading calibration ===")
    data = json.load(path.open('r', encoding='utf-8'))
    
    # Determine calibration method
    calibration_method = data['alignment'].get('calibration_method', 'three_js').lower()
    is_opencv = calibration_method == 'opencv'
    camlogger.logger.info(f"Using calibration method: {calibration_method}")
    
    # First log all raw data
    camlogger.logger.info("")
    camlogger.logger.info("=== Raw Calibration Data ===")
    log_raw_calibration(data)
    
    # Apply all adjustments for both cameras
    Kl, Dl, _, lens_left2cam = adjust_calibration(data, 'left')
    Kr, Dr, _, lens_right2cam = adjust_calibration(data, 'right')
    
    # Get adjusted camera2presentation matrix
    _, _, cam2pres, _ = adjust_calibration(data, 'left')  # Use left camera for presentation matrix
    
    # Log all adjusted data
    camlogger.logger.info("")
    if is_opencv:
        camlogger.logger.info("=== Final Calibration Data (no adjustments needed) ===")
    else:
        camlogger.logger.info("=== Adjusted Calibration Data ===")
    log_adjusted_calibration(Kl, Dl, Kr, Dr, cam2pres, lens_left2cam, lens_right2cam, is_opencv)
    
    # Get reference plane depth
    Z_ref = data['alignment'].get('field_length', 105.0)
    
    camlogger.logger.info("")
    camlogger.logger.info("=== load_calibration.py loaded successfully ===")
    return data, Kl, Dl, Kr, Dr, cam2pres, lens_left2cam, lens_right2cam, is_opencv, Z_ref 