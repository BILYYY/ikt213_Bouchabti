# === Libraries & Functions Used ===
#
# from cv2 (OpenCV):
# - cv2.cvtColor(img, code) with cv2.COLOR_BGR2HSV
#       BGR -> HSV conversion in OpenCV's ranges:
#         H in [0,179], S in [0,255], V in [0,255]
#
# from numpy (np):
# - np.ndarray : image as 3D array (H × W × C)
# - np.empty   : allocate output array
# - Basic math to implement manual conversion

import cv2
import numpy as np

def hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSV using OpenCV.
    OpenCV HSV ranges:
      H: 0..179   (half-degrees)
      S: 0..255
      V: 0..255
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv_manual(image: np.ndarray) -> np.ndarray:
    """
    Manual per-pixel BGR -> HSV conversion, then scaled to OpenCV ranges.
    Steps for one pixel with B,G,R in 0..255:
      1) Normalize: b = B/255, g = G/255, r = R/255
      2) V = max(r,g,b)
      3) m = min(r,g,b), C = V - m
      4) S = 0 if V==0 else C / V
      5) Hue in degrees:
           if C == 0: H = 0
           elif V == r: H = 60 * ((g - b)/C mod 6)
           elif V == g: H = 60 * ((b - r)/C + 2)
           else (V == b): H = 60 * ((r - g)/C + 4)
         if H < 0, add 360.
      6) Map to OpenCV ranges:
           H_cv = round(H / 2)      ∈ [0,179]
           S_cv = round(S * 255)    ∈ [0,255]
           V_cv = round(V * 255)    ∈ [0,255]
    """
    h, w, c = image.shape
    out = np.empty((h, w, 3), dtype=np.uint8)

    # =========================
    # 
    # One pixel [B,G,R] = [0, 255, 0]  (pure green)
    #   b=0, g=1, r=0
    #   V=max=1, m=min=0, C=1
    #   V==g → H = 60*((b-r)/C + 2) = 60*(0 + 2) = 120°
    #   S=C/V=1
    #   OpenCV: H=120/2=60, S=255, V=255 → [60,255,255]
    # =========================

    for i in range(h):
        for j in range(w):
            # Read B,G,R and normalize to 0..1 floats
            B = image[i, j, 0] / 255.0
            G = image[i, j, 1] / 255.0
            R = image[i, j, 2] / 255.0

            V = max(R, G, B)
            m = min(R, G, B)
            C = V - m

            # Saturation
            S = 0.0 if V == 0.0 else (C / V)

            # Hue (in degrees 0..360)
            if C == 0.0:
                H_deg = 0.0
            elif V == R:
                H_deg = 60.0 * (((G - B) / C) % 6.0)
            elif V == G:
                H_deg = 60.0 * (((B - R) / C) + 2.0)
            else:  # V == B
                H_deg = 60.0 * (((R - G) / C) + 4.0)

            if H_deg < 0.0:
                H_deg += 360.0

            # Map to OpenCV HSV ranges
            H_cv = int(round(H_deg / 2.0))      # 0..179
            S_cv = int(round(S * 255.0))        # 0..255
            V_cv = int(round(V * 255.0))        # 0..255

            out[i, j] = [H_cv, S_cv, V_cv]

    return out
