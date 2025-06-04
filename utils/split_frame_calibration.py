import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from enum import Enum
from calibration_logger import (save_intrinsic_calibration, log_intrinsic_results,
                              log_extrinsic_results, save_extrinsic_calibration)

class CalibrationType(Enum):
    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"

class SplitFrameCalibrator:
    def __init__(self, workdir, checkerboard=(7, 10), square_size=1.0):
        """Initialize the calibrator with working directory.
        
        Args:
            workdir: Working directory for output files
            checkerboard: Tuple of (width, height) internal corner count
            square_size: Physical size of each square in millimeters
        """
        self.workdir = Path(workdir)
        self.frames_dir = self.workdir / "frames"
        self.results_dir = self.workdir / "results"
        
        # Create directories if they don't exist
        for dir_path in [self.workdir, self.frames_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Calibration parameters
        self.checkerboard = checkerboard
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points with physical dimensions
        self.objp = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard[0], 
                                   0:self.checkerboard[1]].T.reshape(-1, 2)
        self.objp *= self.square_size  # Scale to physical size

    def find_checkerboard_corners(self, image, frame_name=""):
        """Find checkerboard corners in an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
        # Draw and display the corners for debugging
        debug_image = image.copy()
        if ret:
            cv2.drawChessboardCorners(debug_image, self.checkerboard, corners, ret)
            cv2.putText(debug_image, f"Found corners - {frame_name}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(debug_image, f"No corners - {frame_name}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Resize image for display if too large
        display_width = 800
        if debug_image.shape[1] > display_width:
            scale = display_width / debug_image.shape[1]
            debug_image = cv2.resize(debug_image, None, fx=scale, fy=scale)
        
        cv2.imshow('Checkerboard Detection', debug_image)
        key = cv2.waitKey(1)
        
        return ret, corners if ret else None

    def process_video_for_camera(self, video_path, frame_interval=1, camera="left"):
        """Process video for single camera calibration."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        good_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Split frame and select appropriate half
                height = frame.shape[0]
                mid_height = height // 2
                camera_frame = frame[:mid_height, :] if camera == "left" else frame[mid_height:, :]
                
                # Check for checkerboard
                ret, corners = self.find_checkerboard_corners(camera_frame, 
                                                            f"{camera} {frame_count}")
                if ret:
                    good_frames.append((frame_count, camera_frame, corners))
            
            frame_count += 1
            
        cap.release()
        return good_frames

    def process_video_for_extrinsic(self, video_path, frame_interval=1):
        """Process video for extrinsic calibration."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        good_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Split frame
                height = frame.shape[0]
                mid_height = height // 2
                left_frame = frame[:mid_height, :]
                right_frame = frame[mid_height:, :]
                
                # Check for checkerboard in both views
                left_ret, left_corners = self.find_checkerboard_corners(left_frame, 
                                                                      f"Left {frame_count}")
                right_ret, right_corners = self.find_checkerboard_corners(right_frame, 
                                                                        f"Right {frame_count}")
                
                if left_ret and right_ret:
                    good_frames.append((frame_count, left_corners, right_corners))
            
            frame_count += 1
            
        cap.release()
        return good_frames

    def calibrate_single_camera(self, frames):
        """Calibrate a single camera using collected frames."""
        if len(frames) < 10:
            raise ValueError(f"Not enough frames for calibration (minimum 10 required, got {len(frames)})")
        
        # Prepare calibration data
        objpoints = []
        imgpoints = []
        
        for _, frame, corners in frames:
            objpoints.append(self.objp)
            imgpoints.append(corners)
        
        # Get image size from first frame
        img_size = frames[0][1].shape[:2][::-1]
        
        # Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        
        return ret, mtx, dist

    def calibrate_stereo(self, frames, mtx_left, dist_left, mtx_right, dist_right):
        """Perform stereo calibration using synchronized frames."""
        if len(frames) < 10:
            raise ValueError(f"Not enough stereo frames for calibration (minimum 10 required, got {len(frames)})")
        
        # Prepare calibration data
        objpoints = [self.objp for _ in frames]
        imgpoints_left = [f[1] for f in frames]
        imgpoints_right = [f[2] for f in frames]
        
        # Get image size (assuming all frames are same size)
        img_size = (1920, 1080)  # This should be adjusted based on your video
        
        # Perform stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            img_size, flags=flags
        )
        
        return ret, R, T, E, F

def main():
    parser = argparse.ArgumentParser(description='Camera calibration from split-frame videos')
    parser.add_argument('--type', type=CalibrationType, choices=list(CalibrationType),
                        required=True, help='Type of calibration to perform')
    parser.add_argument('--workdir', type=str, required=True,
                        help='Working directory for temporary and output files')
    parser.add_argument('--frame-interval', type=int, default=1,
                        help='Number of frames to skip between processed frames')
    parser.add_argument('--left-video', type=str,
                        help='Video file for left camera calibration')
    parser.add_argument('--right-video', type=str,
                        help='Video file for right camera calibration')
    parser.add_argument('--stereo-video', type=str,
                        help='Video file for stereo calibration')
    parser.add_argument('--checkerboard', type=str, default='7x10',
                        help='Checkerboard dimensions as WxH internal corners (e.g. 7x10)')
    parser.add_argument('--square-size', type=float, default=20.0,
                        help='Physical size of each checkerboard square in millimeters')
    
    args = parser.parse_args()
    
    # Parse checkerboard dimensions
    try:
        width, height = map(int, args.checkerboard.lower().split('x'))
        checkerboard = (width, height)
    except ValueError:
        print(f"Error: Invalid checkerboard format. Expected WxH (e.g. 7x10), got {args.checkerboard}")
        return
    
    calibrator = SplitFrameCalibrator(args.workdir, checkerboard, args.square_size)
    
    try:
        if args.type == CalibrationType.INTRINSIC:
            if not (args.left_video and args.right_video):
                raise ValueError("Both --left-video and --right-video are required for intrinsic calibration")
            
            print("\nProcessing left camera video...")
            left_frames = calibrator.process_video_for_camera(args.left_video, 
                                                            args.frame_interval, "left")
            print(f"Found {len(left_frames)} good frames for left camera")
            
            print("\nProcessing right camera video...")
            right_frames = calibrator.process_video_for_camera(args.right_video, 
                                                             args.frame_interval, "right")
            print(f"Found {len(right_frames)} good frames for right camera")
            
            # Calibrate both cameras
            print("\nCalibrating cameras...")
            ret_left, mtx_left, dist_left = calibrator.calibrate_single_camera(left_frames)
            ret_right, mtx_right, dist_right = calibrator.calibrate_single_camera(right_frames)
            
            # Log results
            log_intrinsic_results(ret_left, mtx_left, dist_left,
                                ret_right, mtx_right, dist_right)
            
            # Save calibration data
            save_intrinsic_calibration(args.workdir, mtx_left, dist_left, 
                                     mtx_right, dist_right)
            print(f"\nCalibration data saved to {args.workdir}/intrinsic_calibration.json")
            
        else:  # EXTRINSIC
            if not args.stereo_video:
                raise ValueError("--stereo-video is required for extrinsic calibration")
                
            print("\nProcessing stereo video...")
            stereo_frames = calibrator.process_video_for_extrinsic(args.stereo_video, 
                                                                 args.frame_interval)
            print(f"Found {len(stereo_frames)} good stereo frames")
            
            if len(stereo_frames) == 0:
                raise ValueError("No frames found with checkerboard visible in both cameras. " +
                               "Check the video file and checkerboard dimensions.")
            
            # Load intrinsic parameters
            calib_file = Path(args.workdir) / "intrinsic_calibration.json"
            if not calib_file.exists():
                raise FileNotFoundError(
                    f"Intrinsic calibration file not found at {calib_file}. "
                    "Run intrinsic calibration first."
                )
            
            with open(calib_file) as f:
                intrinsic_data = json.load(f)
            
            # Perform stereo calibration
            print("\nPerforming stereo calibration...")
            ret, R, T, E, F = calibrator.calibrate_stereo(
                stereo_frames,
                np.array(intrinsic_data["left_camera_matrix"]),
                np.array(intrinsic_data["left_distortion"]),
                np.array(intrinsic_data["right_camera_matrix"]),
                np.array(intrinsic_data["right_distortion"])
            )
            
            # Log results
            log_extrinsic_results(ret, R, T, E, F)
            
            # Save calibration data
            save_extrinsic_calibration(args.workdir, ret, R, T, E, F)
            print(f"\nCalibration data saved to {args.workdir}/extrinsic_calibration.json")
            
    except ValueError as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 