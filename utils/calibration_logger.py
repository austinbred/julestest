import json
from pathlib import Path
import numpy as np
from camlogger import log_intrinsics, log_distortion, log_extrinsics

def save_intrinsic_calibration(workdir: str, 
                             left_mtx: np.ndarray, 
                             left_dist: np.ndarray,
                             right_mtx: np.ndarray, 
                             right_dist: np.ndarray) -> None:
    """Save intrinsic calibration results to a JSON file."""
    calibration_data = {
        "left_camera_matrix": left_mtx.tolist(),
        "left_distortion": left_dist.tolist(),
        "right_camera_matrix": right_mtx.tolist(),
        "right_distortion": right_dist.tolist()
    }
    
    output_path = Path(workdir) / "intrinsic_calibration.json"
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)

def log_intrinsic_results(ret_left: float, 
                         mtx_left: np.ndarray, 
                         dist_left: np.ndarray,
                         ret_right: float, 
                         mtx_right: np.ndarray, 
                         dist_right: np.ndarray) -> None:
    """Log intrinsic calibration results in a formatted way."""
    print("\nLeft Camera Results:")
    print(f"RMS error: {ret_left}")
    log_intrinsics("Left Camera", mtx_left)
    log_distortion("Left Camera", dist_left)
    
    print("\nRight Camera Results:")
    print(f"RMS error: {ret_right}")
    log_intrinsics("Right Camera", mtx_right)
    log_distortion("Right Camera", dist_right)

def log_extrinsic_results(ret: float, 
                         R: np.ndarray, 
                         T: np.ndarray, 
                         E: np.ndarray, 
                         F: np.ndarray) -> None:
    """Log extrinsic calibration results in a formatted way."""
    print(f"\nStereo Calibration RMS error: {ret}")
    
    # Create 4x4 transformation matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = T.flatten()
    
    log_extrinsics("Stereo", extrinsic_matrix)
    
    print("\nEssential Matrix:")
    print(E)
    print("\nFundamental Matrix:")
    print(F)

def save_extrinsic_calibration(workdir: str,
                              ret: float,
                              R: np.ndarray,
                              T: np.ndarray,
                              E: np.ndarray,
                              F: np.ndarray) -> None:
    """Save extrinsic calibration results to a JSON file."""
    calibration_data = {
        "rms_error": float(ret),
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist()
    }
    
    output_path = Path(workdir) / "extrinsic_calibration.json"
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=4) 