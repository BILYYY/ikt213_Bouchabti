# === Library used ===
# cv2.rotate(src, rotateCode)
# rotateCode:
#   - cv2.ROTATE_90_CLOCKWISE
#   - cv2.ROTATE_180
# (OpenCV handles shape swaps and borders internally.)

import cv2
import numpy as np

def rotation(image: np.ndarray, rotation_angle: int) -> np.ndarray:
    """
    Rotate by 90° clockwise or 180° based on rotation_angle.
    """
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        # If an unsupported angle is given, just return the original (simple behavior)
        return image
