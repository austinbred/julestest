# load_calibration.py
"""
Module: load_calibration.py

Handles loading of calibration JSON, parsing intrinsics, distortion,
 and optional presentation-space extrinsics, with detailed logging.
"""
import json
import numpy as np
from pathlib import Path


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
        Dl (np.ndarray): 4-element left fisheye distortion vector.
        Kr (np.ndarray): 3×3 right camera intrinsic matrix.
        Dr (np.ndarray): 4-element right fisheye distortion vector.
        cam2pres (np.ndarray or None): 4×4 presentation-space transform.
    """
    print("=== load_calibration.py: Loading calibration ===")
    data = json.load(path.open('r', encoding='utf-8'))
    align = data['alignment']

    def parse_K(key: str) -> np.ndarray:
        raw = align[key]
        vals = list(map(float, raw.split(',')))
        print(f"Raw JSON {key} values: {vals}")
        K = np.array(vals, dtype=float).reshape(3, 3)
        print(f"Reshaped {key} (row-major) into matrix:\n{K}")
        print(f"Used {key}: [fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}]")
        return K

    def parse_dist(key: str) -> np.ndarray:
        raw = align[key]
        vals = list(map(float, raw.split(',')))
        print(f"Raw JSON {key} values: {vals}")
        # three.js order: [k1, k2, p1, p2, k3, k4]
        # OpenCV fisheye expects [k1, k2, k3, k4]
        k1 = -vals[0]
        k2 = -vals[1]
        k3 = -vals[4]
        k4 = -vals[5]
        print(f"Extracted {key}: [k1={k1:.2f}, k2={k2:.2f}, k3={k3:.2f}, k4={k4:.2f}]")
        return np.array([k1, k2, k3, k4], dtype=float)

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
        print("Reshaped camera2presentation matrix with labels:")
        for i in range(4):
            row = cam2pres[i]
            labels = [f"m{i}{j}={row[j]:.2f}" for j in range(4)]
            print("[" + ", ".join(labels) + "]")
    else:
        print("No camera2presentation matrix found in JSON.")

    print("=== load_calibration.py loaded successfully ===")
    return data, Kl, Dl, Kr, Dr, cam2pres 