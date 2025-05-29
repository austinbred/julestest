# File: camlogger.py
import numpy as np


def log_intrinsics(name: str, K: np.ndarray) -> None:
    """
    Logs the intrinsic matrix in human-friendly form:
      [ fx=<fx>,  s=<skew>,   cx=<cx> ]
      [ <c10>,  fy=<fy>,  cy=<cy> ]
      [ <c20>,   <c21>,   <c22> ]

    Cells marked "error=<value>" indicate deviations from expected zero or one.

    Parameters:
    - name: label for the matrix (e.g., 'LeftCam')
    - K: 3x3 intrinsic matrix
    """
    K = np.array(K, dtype=float)
    # Extract standard intrinsics
    fx, s, cx = K[0,0], K[0,1], K[0,2]
    v10, fy, cy = K[1,0], K[1,1], K[1,2]
    v20, v21, v22 = K[2,0], K[2,1], K[2,2]

    # Determine error flags for predefined zeros/ones
    c10 = f"0.000000" if abs(v10) < 1e-6 else f"error={v10:0.6f}"
    c20 = f"0.000000" if abs(v20) < 1e-6 else f"error={v20:0.6f}"
    c21 = f"0.000000" if abs(v21) < 1e-6 else f"error={v21:0.6f}"
    c22 = f"1.000000" if abs(v22 - 1.0) < 1e-6 else f"error={v22:0.6f}"

            # Prepare labeled fields
    col00 = f"fx={fx:0.6f}"; col01 = f"s={s:0.6f}"; col02 = f"cx={cx:0.6f}"
    col10 = c10;              col11 = f"fy={fy:0.6f}"; col12 = f"cy={cy:0.6f}"
    col20 = c20;              col21 = c21;              col22 = c22

    # Pad each column entry to 12 characters, right-aligned
    col00 = f"{col00:>12}"; col01 = f"{col01:>12}"; col02 = f"{col02:>12}"
    col10 = f"{col10:>12}"; col11 = f"{col11:>12}"; col12 = f"{col12:>12}"
    col20 = f"{col20:>12}"; col21 = f"{col21:>12}"; col22 = f"{col22:>12}"

    # Construct rows with aligned columns
    row0 = f"[ {col00}, {col01}, {col02} ]"
    row1 = f"[ {col10}, {col11}, {col12} ]"
    row2 = f"[ {col20}, {col21}, {col22} ]"

    # Ensure name is not empty is not empty is not empty
    name_str = name if name else "Camera"
        # Print a blank line for separation
    print()
    # Print the matrix
    print(f"{name_str} Intrinsics:")
    print(row0)
    print(row1)
    print(row2)

# Example usage in step04_rigwarp.py:
# import camlogger
# camlogger.log_intrinsics('LeftCam', K_src) in step04_rigwarp.py:
# import camlogger
# camlogger.log_intrinsics('LeftCam', K_src)

