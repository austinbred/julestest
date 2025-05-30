import numpy as np

def log_distortion(name: str, D: np.ndarray) -> None:
    """
    Logs the distortion matrix with :
      [ d00=<d00>, d01=<d01>, d02=<d02>, d03=<d03> ]
    """

    D = np.array(D, dtype=float)
    d = D.flatten()

    # Print rotation rows
    print()
    print(f"{name} Distortion:")
    print(f"[ d00={d[0]:12.6f}, d01={d[1]:12.6f}, d02={d[2]:12.6f}, d03={d[3]:12.6f}]")


def log_intrinsics(name: str, K: np.ndarray) -> None:
    """
    Logs the intrinsic matrix with fixed six-decimal formatting:
      [ fx=<fx>,  s=<skew>,   cx=<cx> ]
      [ <c10>,  fy=<fy>,  cy=<cy> ]
      [ <c20>,   <c21>,   <c22> ]

    All values are shown with six decimal places for clarity.
    """
    K = np.array(K, dtype=float)
    fx, s, cx = K[0,0], K[0,1], K[0,2]
    v10, fy, cy = K[1,0], K[1,1], K[1,2]
    v20, v21, v22 = K[2,0], K[2,1], K[2,2]

    # Print aligned rows with six decimal places
    print()
    print(f"{name} Intrinsics:")
    print(f"[ fx={ fx:12.6f},   s={  s:12.6f},  cx={cx:12.6f} ]")
    print(f"[    {v10:12.6f},  fy={ fy:12.6f},     {cy:12.6f} ]")
    print(f"[    {v20:12.6f},     {v21:12.6f},     {v22:12.6f} ]")


def log_extrinsics(name: str, E: np.ndarray) -> None:
    """
    Logs the extrinsic matrix with rotation and translation labels in six-decimal format:
      [ r00=<r00>, r01=<r01>, r02=<r02> ]
      [ r10=<r10>, r11=<r11>, r12=<r12> ]
      [ r20=<r20>, r21=<r21>, r22=<r22> ]
      Translation: [ tx=<tx>, ty=<ty>, tz=<tz> ]
    """
    E = np.array(E, dtype=float)
    R = E[:4, :4]
    t = E[:3, 3]

    # Print rotation rows
    print()
    print(f"{name} Extrinsics:")
    print(f"[ r00={R[0,0]:12.6f}, r01={R[0,1]:12.6f}, r02={R[0,2]:12.6f}, tx={R[0,3]:12.6f} ]")
    print(f"[ r10={R[1,0]:12.6f}, r11={R[1,1]:12.6f}, r12={R[1,2]:12.6f}, ty={R[1,3]:12.6f} ]")
    print(f"[ r20={R[2,0]:12.6f}, r21={R[2,1]:12.6f}, r22={R[2,2]:12.6f}, tz={R[2,3]:12.6f} ]")
    print(f"[ 0={R[3,0]:12.6f}, 0={R[3,1]:12.6f}, 0={R[3,2]:12.6f}, 1={R[3,3]:12.6f} ]")

    # Print translation
    tx, ty, tz = t
    print(f"Translation: [ tx={tx:0.6f}, ty={ty:0.6f}, tz={tz:0.6f} ]")

