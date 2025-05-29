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
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Step modules
import step01_split
import step02_expand
import step03_undistort
import step04_rigwarp
import step05_canvas
import step06_warp
import step07_presentation
import calibration


def main():
    parser = argparse.ArgumentParser(description="VEO panoramic pipeline")
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--input',       required=True, type=Path)
    parser.add_argument('--output',      required=True, type=Path)
    parser.add_argument('--workdir',     required=True, type=Path)
    args = parser.parse_args()

    # Load calibration using calibration.py utility
    data, Kl, Dl, Kr, Dr, cam2pres = calibration.load_calibration(args.calibration)

    # Prepare base working directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_workdir = args.workdir.resolve() / f"job_{timestamp}"
    base_workdir.mkdir(parents=True, exist_ok=True)

    # Reference depth (e.g., field width)
    Z_ref = float(data.get('field_width', 1.0))
    print(f"Using reference plane depth Z_ref = {Z_ref:.5f}")

    # Field length for warp (optional)
    field_length = float(data.get('field_length', Z_ref))

    # Video I/O setup
    cap = cv2.VideoCapture(str(args.input))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    # Parse extrinsics manually
    lens_left2camera  = np.array(list(map(float, data['alignment']['lens_left2camera'].split(','))), dtype=float).reshape((4,4))
    lens_right2camera = np.array(list(map(float, data['alignment']['lens_right2camera'].split(','))), dtype=float).reshape((4,4))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a unique working directory for this frame
        frame_dir = base_workdir / f"frame{idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save original frame PNG
        frame_path = frame_dir / f"frame{idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)

        # Step 01: split
        l1, r1 = step01_split.split_frame(frame, frame_dir, idx)

        # Step 02: expand each half into canvas
        l2 = step02_expand.expand_image(l1, frame_dir)
        r2 = step02_expand.expand_image(r1, frame_dir)

        # Step 03: undistort fisheye
        l3 = step03_undistort.undistort_fisheye(l2, Kl, Dl, frame_dir)
        r3 = step03_undistort.undistort_fisheye(r2, Kr, Dr, frame_dir)

        # Step 04: warp each lens into rig frame
        l4 = step04_rigwarp.warp_to_rig(l3, Kl, lens_left2camera,  np.eye(3), Z_ref, frame_dir, idx, 'left')
        r4 = step04_rigwarp.warp_to_rig(r3, Kr, lens_right2camera, np.eye(3), Z_ref, frame_dir, idx, 'right')

        # Step 05: stitch rig-space images
        canvas = step05_canvas.compose_canvas(l4, r4, 0, frame_dir, idx)

        # Step 06: bird's-eye (optional)
        be = step06_warp.warp_birdseye(canvas, data, frame_dir, idx)

        # Step 07: presentation view (using camera2presentation matrix)
        pres = step07_presentation.apply_presentation(be, data, frame_dir, idx)

        # Write to output video
        out_frame = cv2.cvtColor(np.array(pres), cv2.COLOR_RGB2BGR)
        out.write(out_frame)

        idx += 1

    cap.release()
    out.release()

if __name__ == '__main__':
    main()

