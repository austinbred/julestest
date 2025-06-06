"""
Module: step02_expand.py

Expands images to a larger canvas and adjusts intrinsic matrices accordingly.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import camlogger
from camlogger import log_intrinsics

# Canvas dimensions for expanded images
CANVAS_WIDTH = 4096
CANVAS_HEIGHT = 2048

def adjust_principal_point(K: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    """
    Adjusts the principal point in the intrinsic matrix based on image position in canvas.
    
    Args:
        K: Original intrinsic matrix
        x_offset: Horizontal offset of image in canvas
        y_offset: Vertical offset of image in canvas
        
    Returns:
        K_adj: Adjusted intrinsic matrix
    """
    K_adj = K.copy()
    K_adj[0, 2] += x_offset  # cx
    K_adj[1, 2] += y_offset  # cy
    
    camlogger.logger.info(f"Principal point adjustment: x_offset={x_offset}, y_offset={y_offset}")
    return K_adj

def expand_to_canvas(img: np.ndarray, K: np.ndarray, workdir: Path, idx: int, side: str):
    """
    Places an image onto a larger canvas and adjusts intrinsic matrix.
    
    Args:
        img (np.ndarray): Input image array
        K (np.ndarray): Intrinsic matrix to adjust
        workdir (Path): Directory to save output
        idx (int): Frame index for output filenames
        side (str): Camera side ('left' or 'right')
        
    Returns:
        tuple: (expanded_img, K_expanded)
            - expanded_img: Canvas-expanded image array
            - K_expanded: Canvas-adjusted intrinsic matrix
    """
    camlogger.logger.info("")
    camlogger.logger.info("=== step02_expand.py: Starting canvas expansion and principal point adjustment ===")
    
    # Log canvas dimensions
    camlogger.logger.info("")
    camlogger.logger.info("=== Canvas Dimensions ===")
    camlogger.logger.info(f"Target canvas dimensions: {CANVAS_WIDTH}×{CANVAS_HEIGHT}")
    
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    w, h = img_pil.size
    camlogger.logger.info(f"Input image size: width={w}, height={h}")
    
    # Create canvas
    canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0))
    camlogger.logger.info(f"Created canvas: width={CANVAS_WIDTH}, height={CANVAS_HEIGHT}")
    
    # Calculate horizontal position (centered)
    x_offset = (CANVAS_WIDTH - w) // 2
    
    # Calculate vertical position (centered in respective half)
    y_offset = (CANVAS_HEIGHT - h) // 2
    
    # Paste image onto canvas
    canvas.paste(img_pil, (x_offset, y_offset))
    camlogger.logger.info(f"Placed image at position: x={x_offset}, y={y_offset}")
    
    # Save output
    out_path = workdir / f"02_{side}.png"
    canvas.save(out_path)
    camlogger.logger.info(f"Saved expanded image: {out_path}")
    
    # Adjust intrinsic matrix for both x and y offsets
    camlogger.logger.info("")
    camlogger.logger.info("Adjusting intrinsic matrix for canvas position:")
    log_intrinsics(K, "Original (pre-canvas)")
    
    # Adjust principal point for both x and y offsets
    K_expanded = adjust_principal_point(K, x_offset, y_offset)
    log_intrinsics(K_expanded, "Adjusted (post-canvas)")
    
    camlogger.logger.info("")
    camlogger.logger.info("=== step02_expand.py completed successfully ===")
    
    # Convert back to numpy array
    canvas_array = np.array(canvas)
    
    return canvas_array, K_expanded
