#!/usr/bin/env python3
"""
pipeline.py

Main pipeline orchestrating VEO rig to panorama presentation:
  01: split stereoscopic frame
  02: expand to canvas
  03: fisheye undistort
  04: warp lenses into rig frame (with translation)
  05: stitch rig-space images
  06: (optional) bird's-eye warp
  07: presentation view
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import camlogger

import step01_split
import step02_expand
import step03_undistort
import step04_rigwarp
import step05_stitch
import step06_warp
import step07_presentation
import load_calibration

def parse_args():
    parser = argparse.ArgumentParser(description="Process VEO stereo frames through calibration pipeline")
    parser.add_argument('--calibration', type=str, required=True, help='Path to calibration JSON')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--workdir', type=str, required=True, help='Working directory for intermediate files')
    parser.add_argument('--output', type=str, help='Output directory for final frames')
    parser.add_argument('--frames', type=int, help='Number of frames to process (default: all)')
    return parser.parse_args()

def process_frame(frame: np.ndarray, workdir: Path, calib_path: str):
    """
    Process a single stereo frame through the calibration pipeline.
    
    Args:
        frame (np.ndarray): BGR frame from OpenCV
        workdir (Path): Directory to save intermediate outputs
        calib_path (str): Path to calibration JSON file
        
    Returns:
        tuple: (left_exp_path, right_exp_path, K_left_exp, K_right_exp)
            - left_exp_path: Path to expanded left image
            - right_exp_path: Path to expanded right image
            - K_left_exp: Canvas-adjusted left intrinsic matrix
            - K_right_exp: Canvas-adjusted right intrinsic matrix
    """
    # Log pipeline constants
    camlogger.logger.info("")
    camlogger.logger.info("=== Pipeline Constants ===")
    camlogger.logger.info(f"Original sensor dimensions: {step01_split.SENSOR_WIDTH}×{step01_split.SENSOR_HEIGHT}")
    camlogger.logger.info(f"Canvas dimensions: {step02_expand.CANVAS_WIDTH}×{step02_expand.CANVAS_HEIGHT}")
    camlogger.logger.info("")

    # Load calibration matrices
    data, Kl, Dl, Kr, Dr, cam2pres, lens_left2cam, lens_right2cam, is_opencv = load_calibration.load_calibration(Path(calib_path))
    
    # Split frame into left/right halves
    left_path, right_path, K_left, K_right = step01_split.split_frame(
        frame, workdir, Kl, Kr
    )
    
    # Expand images to canvas
    left_exp_path, K_left_exp = step02_expand.expand_image(left_path, workdir, K_left)
    right_exp_path, K_right_exp = step02_expand.expand_image(right_path, workdir, K_right)
    
    return left_exp_path, right_exp_path, K_left_exp, K_right_exp

def main():
    parser = argparse.ArgumentParser(description='Process VEO panoramic video frames')
    parser.add_argument('--calibration', required=True, help='Path to calibration JSON')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--workdir', required=True, help='Path to working directory')
    args = parser.parse_args()
    
    # Create working directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workdir = Path(args.workdir) / f'job_{timestamp}'
    workdir.mkdir(parents=True, exist_ok=True)
    camlogger.logger.info(f"Created working directory: {workdir}")
    
    # Load calibration data
    data, Kl, Dl, Kr, Dr, cam2pres, lens_left2cam, lens_right2cam, is_opencv, Z_ref = load_calibration.load_calibration(Path(args.calibration))
    
    # Open input video
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")
    camlogger.logger.info(f"Opened input video: {args.input}")
    
    # Parse lens-to-camera extrinsic matrices
    camlogger.logger.info("Parsed lens-to-camera extrinsic matrices")
    
    # Process first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")
    
    # Process frames
    idx = 0
    while ret:
        camlogger.logger.info(f"\nProcessing frame {idx:06d}")
        
        # Create frame directory
        frame_dir = workdir / f'frame_{idx:06d}'
        frame_dir.mkdir(parents=True, exist_ok=True)
        camlogger.logger.info(f"Created frame directory: {frame_dir}")
        
        # Save original frame
        frame_path = frame_dir / '00_original.png'
        cv2.imwrite(str(frame_path), frame)
        camlogger.logger.info(f"Saved original frame: {frame_path}")
        
        # Log pipeline constants
        camlogger.logger.info("\n=== Pipeline Constants ===")
        camlogger.logger.info("Original sensor dimensions: 3840×2160")
        camlogger.logger.info("Canvas dimensions: 4096×2048")
        
        # Step 1: Split frame and scale intrinsics
        l1, r1, K_left_split, K_right_split = step01_split.split_frame(frame, Kl, Kr, frame_dir, idx)
        
        # Step 2: Expand to canvas and adjust principal points
        l2, K_left_exp = step02_expand.expand_to_canvas(l1, K_left_split, frame_dir, idx, 'left')
        r2, K_right_exp = step02_expand.expand_to_canvas(r1, K_right_split, frame_dir, idx, 'right')
        
        # Step 3: Undistort
        l3 = step03_undistort.undistort_frame(l2, K_left_exp, Dl, frame_dir, idx, 'left', is_opencv)
        r3 = step03_undistort.undistort_frame(r2, K_right_exp, Dr, frame_dir, idx, 'right', is_opencv)
        
        # Step 4: Warp to rig
        l4 = step04_rigwarp.warp_to_rig(l3, K_left_exp, lens_left2cam, np.eye(3), Z_ref, frame_dir, idx, 'left')
        r4 = step04_rigwarp.warp_to_rig(r3, K_right_exp, lens_right2cam, np.eye(3), Z_ref, frame_dir, idx, 'right')
        
        # Read next frame
        ret, frame = cap.read()
        idx += 1
    
    camlogger.logger.info("\nProcessing complete!")
    cap.release()

if __name__ == '__main__':
    main()

