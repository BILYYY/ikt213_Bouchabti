# === Library used ===
# cv2.cvtColor(img, code)            : color-space conversion (BGR↔HSV)
# OpenCV HSV ranges: H∈[0,179], S∈[0,255], V∈[0,255]
# We shift ONLY the Hue channel by `hue` and wrap with mod 180.

import cv2
import numpy as np

def hue_shifted(image: np.ndarray, emptyPictureArray: np.ndarray, hue: int) -> np.ndarray:
    """
    Shift the RGB color values by +hue and handle wrapping at 255/0.
    """
    shifted = (image.astype(np.int16) + hue) % 256
    shifted = shifted.astype(np.uint8)
    
    emptyPictureArray[:] = shifted
    return emptyPictureArray
