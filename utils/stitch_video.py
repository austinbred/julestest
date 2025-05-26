#!/usr/bin/env python3
"""
Affine-based panorama stitcher for top/bottom stereo video.

1. Samples ORB matches across frames to estimate a robust 2×3 affine transform A from bottom→top.
2. Applies the fixed affine warp to all bottom halves, computes a joint canvas bounds, and stitches the warped bottom and original top into a seamless panorama.

Usage:
   pip install opencv-python
   chmod +x stitch_video.py
   ./stitch_video.py input_stack.mp4 output_pano.mp4
"""
import cv2
import numpy as np
import sys

def calibrate_affine(video_path, sample_every=10, ratio_thresh=0.75, min_matches=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orb = cv2.ORB_create(5000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    pts_bot, pts_top = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            h, w = frame.shape[:2]
            half = h // 2
            top = frame[:half, :]
            bot = frame[half:, :]

            k1,d1 = orb.detectAndCompute(top, None)
            k2,d2 = orb.detectAndCompute(bot, None)
            if d1 is None or d2 is None:
                frame_idx += 1; continue

            matches = bf.knnMatch(d2, d1, k=2)
            for m,n in matches:
                if m.distance < ratio_thresh * n.distance:
                    pts_bot.append(k2[m.queryIdx].pt)
                    pts_top.append(k1[m.trainIdx].pt)
        frame_idx += 1
    cap.release()

    if len(pts_bot) < min_matches:
        raise RuntimeError(f"Not enough matches ({len(pts_bot)}) for affine calibration.")

    pts_bot = np.float32(pts_bot)
    pts_top = np.float32(pts_top)
    A, inliers = cv2.estimateAffinePartial2D(
        pts_bot, pts_top,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    if A is None:
        raise RuntimeError("Affine estimation failed.")
    return A


def apply_affine(video_in, video_out, A):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    # read first frame to get sizes
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")
    h, w = frame0.shape[:2]
    half = h // 2

    # compute warped corners of bottom half
    bot_corners = np.float32([[0,0],[w,0],[0,half],[w,half]])
    warped = cv2.transform(np.array([bot_corners]), A)[0]
    # top half corners remain original
    top_corners = np.float32([[0,0],[w,0],[0,half],[w,half]])
    all_pts = np.vstack((warped, top_corners))
    xmin, ymin = np.floor(all_pts.min(axis=0) - 0.5).astype(int)
    xmax, ymax = np.ceil(all_pts.max(axis=0) + 0.5).astype(int)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    # translation to shift into positive coords
    trans = np.array([[1, 0, -xmin], [0, 1, -ymin]])
    M = trans.dot(np.vstack((A, [0,0,1])))[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out, fourcc, fps, (canvas_w, canvas_h))
    # rewind to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        top = frame[:half, :]
        bot = frame[half:, :]

        # warp bottom into canvas
        warp_bot = cv2.warpAffine(bot, M, (canvas_w, canvas_h))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=frame.dtype)

        # paste warped bottom and original top
        canvas_y = -ymin
        canvas_x = -xmin
        # top sits at (canvas_x, canvas_y)
        canvas[canvas_y:canvas_y+half, canvas_x:canvas_x+w] = top
        # warped bottom already placed by warpAffine
        # composite any non-zero warp_bot pixels (optional alpha blend)
        mask = (warp_bot != 0)
        canvas[mask] = warp_bot[mask]

        out.write(canvas)

    cap.release()
    out.release()


def main():
    if len(sys.argv) != 3:
        print("Usage: stitch_video.py input_stack.mp4 output_pano.mp4")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    print("[*] Calibrating affine transform...")
    A = calibrate_affine(inp, sample_every=10)
    print("[*] Applying affine warp and stitching...")
    apply_affine(inp, outp, A)
    print(f"Done → {outp}")

if __name__ == "__main__":
    main()

