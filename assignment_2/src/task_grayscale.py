# === Libraries & Functions Used ===
#
# from cv2 (OpenCV):
# - cv2.cvtColor(img, code)
#       Convert image color space.
#       Inputs:  img (ndarray), code (e.g., cv2.COLOR_BGR2GRAY)
#       Output:  converted image (ndarray)
#
# from numpy (np):
# - np.ndarray : image as 3D array (H × W × C)
# - np.empty   : allocate output array
# - np.uint8   : pixels in range 0..255

import cv2
import numpy as np

def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Output is a 2D array (H × W), dtype uint8 (0..255).
    """
    # OpenCV reads color as BGR;
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def grayscale_manual(image: np.ndarray) -> np.ndarray:
    """
    grayscale conversion (per pixel) using the BT.601 luma formula:
      gray = 0.114*B + 0.587*G + 0.299*R
    Why these weights? Human vision is most sensitive to Green, then Red, least to Blue.
    """
    h, w, c = image.shape            # c should be 3 for B,G,R
    out = np.empty((h, w), dtype=np.uint8)  # 2D output (no channels)

    # =========================
    # Mini example to visualize:
    # Suppose a tiny 2×3 color image with per-pixel triplets [B,G,R]:
    #
    # i\j      0           1           2
    #  0    [B0,G0,R0]  [B1,G1,R1]  [B2,G2,R2]
    #  1    [B3,G3,R3]  [B4,G4,R4]  [B5,G5,R5]
    #
    # After grayscale, each pixel becomes a single number (0..255):
    #
    # i\j    0      1      2
    #  0   g0     g1     g2     where gk = round(0.114*Bk + 0.587*Gk + 0.299*Rk)
    #  1   g3     g4     g5
    # =========================

    for i in range(h):          # loop rows
        for j in range(w):      # loop columns
            # image[i, j] is [B, G, R] (uint8). Convert to ints for safe math.
            B = int(image[i, j, 0])
            G = int(image[i, j, 1])
            R = int(image[i, j, 2])

            # Weighted sum (BT.601 luma). round() avoids systematic bias.
            gray_val = int(round(0.114 * B + 0.587 * G + 0.299 * R))

            # Store into output (uint8 automatically clamps 0..255)
            out[i, j] = gray_val

    return out
