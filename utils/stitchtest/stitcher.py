#!/usr/bin/env python3
"""
Rotation model images stitcher.

Usage:
    python stitcher.py img1 img2 [...imgN] [flags]
"""
import sys
import argparse
import cv2
import numpy as np
from cv2 import detail

def parse_args():
    parser = argparse.ArgumentParser(description='Rotation model images stitcher.')
    parser.add_argument('images', nargs='+', help='Input images')
    parser.add_argument('--preview', action='store_true', help='Run stitching in preview mode')
    parser.add_argument('--result', dest='result_name', default='result.jpg', help='Output result image file name')
    parser.add_argument('--features', choices=['orb','akaze','sift','surf'], default='surf', help='Feature type')
    # (other flags omitted for brevity)
    args = parser.parse_args()
    if args.preview:
        args.compose_megapix = 0.6
    return args


def main():
    args = parse_args()
    img_names = args.images
    if len(img_names) < 2:
        print('Need at least two images')
        sys.exit(1)

    # Attempt high-level API
    stitcher = None
    try:
        stitcher = cv2.Stitcher_create()  # OpenCV 4.x
    except AttributeError:
        try:
            stitcher = cv2.createStitcher(False)  # older API fallback
        except Exception:
            stitcher = None
    if stitcher:
        imgs = []
        for name in img_names:
            img = cv2.imread(name)
            if img is None:
                print(f"Can't open {name}")
                sys.exit(1)
            imgs.append(img)
        status, pano = stitcher.stitch(imgs)
        if status == cv2.Stitcher_OK:
            cv2.imwrite(args.result_name, pano)
            print('Stitching succeeded with high-level API')
            return
        else:
            print(f'High-level Stitcher failed (status={status}), falling back to detailed pipeline')

    # Detailed pipeline (manual) below
    # ... existing manual pipeline code omitted for brevity ...
    print('Detailed pipeline is not yet implemented in this version.')

if __name__ == '__main__':
    main()

