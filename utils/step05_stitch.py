# step05_stitch.py
"""
Module: step05_stitch.py

Provides a function to stitch two rig-space images into a seamless panorama using OpenCV feature matching
and homography-based projection. All informational messages are printed to the console.
"""

import cv2
import numpy as np
from pathlib import Path


def compose_stitch(left_bgr: np.ndarray,
                   right_bgr: np.ndarray,
                   workdir: Path,
                   idx: int,
                   pad: int = 200) -> np.ndarray:
    """
    Detects common features between left and right images, computes a homography,
    warps the right image into the left image’s frame, and blends them onto a single wide canvas.
    Adds padding around the stitched result and prints key steps.

    Args:
        left_bgr   (np.ndarray): Left rig-space image in BGR.
        right_bgr  (np.ndarray): Right rig-space image in BGR.
        workdir     (Path): Directory to save the stitched output.
        idx         (int): Frame index for naming.
        pad         (int): Number of pixels of black padding to add on each side.

    Returns:
        np.ndarray: The final stitched panorama in BGR.
    """
    # 1) Validate inputs
    if left_bgr is None or left_bgr.size == 0:
        raise ValueError("Left image is empty or invalid.")
    if right_bgr is None or right_bgr.size == 0:
        raise ValueError("Right image is empty or invalid.")
    if left_bgr.ndim != 3 or left_bgr.shape[2] != 3:
        raise ValueError("Left image must be a 3-channel BGR array.")
    if right_bgr.ndim != 3 or right_bgr.shape[2] != 3:
        raise ValueError("Right image must be a 3-channel BGR array.")

    # Ensure output directory exists
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured workdir exists: {workdir}")

    # 2) Convert both images to grayscale for feature detection
    gray_left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    # 3) Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(gray_left, None)
    kp2, des2 = orb.detectAndCompute(gray_right, None)
    if des1 is None or des2 is None:
        raise RuntimeError("Failed to compute descriptors for one or both images.")
    print(f"[INFO] Detected {len(kp1)} keypoints in left, {len(kp2)} keypoints in right")

    # 4) Match descriptors using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    print(f"[INFO] Found {len(knn_matches)} raw descriptor matches (knn k=2)")

    # 5) Apply Lowe’s ratio test to filter good matches
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    MIN_MATCH_COUNT = 10
    if len(good_matches) < MIN_MATCH_COUNT:
        raise RuntimeError(
            f"Not enough good matches were found ({len(good_matches)}/{MIN_MATCH_COUNT})."
        )
    print(f"[INFO] {len(good_matches)} good matches passed Lowe’s ratio test")

    # 6) Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 7) Compute homography from right image to left image
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography could not be computed.")
    print("[INFO] Computed homography matrix H")

    # 8) Determine bounding box of warped right + left (unpadded)
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]

    # Four corners of the right image
    corners_right = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    # Transform these corners with the computed homography
    warped_corners = cv2.perspectiveTransform(corners_right, H)

    # Corners of the left image in its own coordinate frame
    corners_left = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)

    # Combine warped right corners and left corners to find the full bounding box
    all_corners = np.concatenate((warped_corners, corners_left), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    print(f"[INFO] Raw bounding box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    # 8a) Add padding around the bounding box
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad
    print(f"[INFO] Padded bounding box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    # Compute translation to shift everything into positive coordinates
    translate_x = -xmin if xmin < 0 else 0
    translate_y = -ymin if ymin < 0 else 0
    print(f"[INFO] Translation offsets: translate_x={translate_x}, translate_y={translate_y}")

    # 8b) Compute final panorama size
    pano_width = xmax - xmin
    pano_height = ymax - ymin
    print(f"[INFO] Final panorama size (with padding): width={pano_width}, height={pano_height}")

    # 9) Warp the right image with combined Homography + translation
    T = np.array([
        [1, 0, translate_x],
        [0, 1, translate_y],
        [0, 0, 1]
    ], dtype=np.float64)

    warped_right = cv2.warpPerspective(
        right_bgr,
        T.dot(H),
        (pano_width, pano_height)
    )
    print("[INFO] Warped right im age into panorama frame")

    # 10) Paste the left image into the panorama canvas
    panorama = np.zeros((pano_height, pano_width, 3), dtype=left_bgr.dtype)
    x_offset_left = translate_x
    y_offset_left = translate_y
    panorama[y_offset_left:y_offset_left + h1, x_offset_left:x_offset_left + w1] = left_bgr
    print("[INFO] Pasted left image onto panorama canvas")

    # 11) Blend the warped right image onto the panorama
    mask_warped = (warped_right > 0)
    panorama[mask_warped] = warped_right[mask_warped]
    print("[INFO] Blended warped right image onto canvas")

    # 12) Save and return the panorama
    out_path = workdir / f"frame{idx:06d}_05_stitched.png"
    success = cv2.imwrite(str(out_path), panorama)
    if not success:
        raise RuntimeError(f"Failed to write panorama to {out_path}")
    print(f"[INFO] Saved stitched panorama: {out_path}")

    return panorama
