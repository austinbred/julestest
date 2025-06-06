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
from camlogger import log_intrinsics, log_distortion


def load_calibration(path: Path):
    """
    Load and parse intrinsics (K), distortion (D), and optional camera2presentation matrix.
    Logs header and detailed parse information including raw JSON values
    and final matrices/vectors used by OpenCV.

    Args:
        path (Path): Path to calibration JSON file.

    Returns:
        data (dict): Full JSON data.
        Kl (np.ndarray): 3×3 left camera intrinsic matrix.
        Dl (np.ndarray): Distortion coefficients (5 for OpenCV, 4 for fisheye).
        Kr (np.ndarray): 3×3 right camera intrinsic matrix.
        Dr (np.ndarray): Distortion coefficients (5 for OpenCV, 4 for fisheye).
        cam2pres (np.ndarray or None): 4×4 presentation-space transform.
        is_opencv (bool): True if using OpenCV calibration format, False for Three.js format.
    """
    print("=== load_calibration.py: Loading calibration ===")
    data = json.load(path.open('r', encoding='utf-8'))
    align = data['alignment']
    
    # Determine calibration method
    calibration_method = align.get('calibration_method', 'three_js').lower()
    is_opencv = calibration_method == 'opencv'
    print(f"Using calibration method: {calibration_method}")

    def parse_K(key: str) -> np.ndarray:
        raw = align[key]
        vals = list(map(float, raw.split(',')))
        print(f"Raw JSON {key} values: {vals}")
        K = np.array(vals, dtype=float).reshape(3, 3)
        log_intrinsics(K, key)
        return K

    def parse_dist(key: str) -> np.ndarray:
        raw = align[key]
        vals = list(map(float, raw.split(',')))
        print(f"Raw JSON {key} values: {vals}")
        
        if is_opencv:
            # OpenCV format: Use all 5 coefficients (k1,k2,k3,p1,p2)
            print(f"Using OpenCV format - keeping all 5 distortion coefficients")
            D = np.array(vals[:5], dtype=float)
            log_distortion(key, D, is_opencv)
            return D
        else:
            # Three.js format: [k1, k2, p1, p2, k3, k4, k5, k6, 0, 0, 0, 0, 0, 0]
            # Convert to OpenCV fisheye format: [k1, k2, k3, k4]
            # Note: Three.js uses opposite sign convention for k1,k2,k3,k4
            k1, k2 = -vals[0], -vals[1]  # First two values
            k3, k4 = -vals[6], -vals[7]  # Seventh and eighth values (k5, k6 in Three.js)
            D = np.array([k1, k2, k3, k4], dtype=float)
            log_distortion(key, D, is_opencv)
            return D

    # Parse intrinsics and distortion
    Kl = parse_K('intrinsic_left')
    Kr = parse_K('intrinsic_right')
    Dl = parse_dist('distortion_left')
    Dr = parse_dist('distortion_right')

    # Optional presentation-space transform
    cam2pres = None
    if 'camera2presentation' in align:
        raw = align['camera2presentation']
        vals = list(map(float, raw.split(',')))
        print(f"Raw JSON camera2presentation values: {[f'{v:.2f}' for v in vals]}")
        cam2pres = np.array(vals, dtype=float).reshape((4, 4))
        log_intrinsics(cam2pres, "camera2presentation")
    else:
        print("No camera2presentation matrix found in JSON.")

    print("=== load_calibration.py loaded successfully ===")
    return data, Kl, Dl, Kr, Dr, cam2pres, is_opencv 