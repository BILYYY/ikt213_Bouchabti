# === Library used ===
# cv2.cvtColor(img, code)            : color-space conversion (BGR↔HSV)
# OpenCV HSV ranges: H∈[0,179], S∈[0,255], V∈[0,255]
# We shift ONLY the Hue channel by `hue` and wrap with mod 180.

import cv2
import numpy as np

def hue_shifted(image: np.ndarray, emptyPictureArray: np.ndarray, hue: int) -> np.ndarray:
    """
    Shift the image hue by +hue (OpenCV Hue units), write result into emptyPictureArray.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   # BGR → HSV
    H, S, V = cv2.split(hsv)                       # separate channels

    # Shift hue and wrap in [0,179] (OpenCV hue range)
    H = (H.astype(np.int16) + int(hue)) % 180
    H = H.astype(np.uint8)

    hsv_shifted = cv2.merge([H, S, V])
    bgr_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)  # HSV → BGR

    # Write into the provided emptyPictureArray (same shape/type as input)
    emptyPictureArray[:] = bgr_shifted
    return emptyPictureArray
