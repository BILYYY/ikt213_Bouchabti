# === Library used ===
#
# here in smoothing in opencv i can use different filter not just gaussian
# - cv2.blur(img, (k,k))           → box filter (average blur)
# - cv2.GaussianBlur(img,(k,k),0) → gaussian blur
# - cv2.medianBlur(img,k)         → median filter
#
# they all blur but results not the same
#
# box filter → all pixels same average, kills edges
# gaussian   → center pixel gets more weight, smoother keep edges a bit
# median     → pick the middle value, good for salt noise, edges safe
#
# === Example with 3x3 kernel and patch:
# input patch:
# 1 2 3
# 4 5 6
# 7 8 9
#
# BOX FILTER (all average):
# (1+2+3+4+5+6+7+8+9)/9 = 5
# result:
# 5 5 5
# 5 5 5
# 5 5 5
#
# GAUSSIAN (kernel = [[1 2 1],[2 4 2],[1 2 1]] /16)
# weighted sum = 80/16 = 5 for center
# but not all values same, near edges diff
# result approx:
# 3 4 4
# 5 5 6
# 6 7 7
#
# MEDIAN:
# sort all [1..9] median=5 → center=5
# edge pixels take median of their 3x3 window
# result approx:
# 2 3 3
# 4 5 6
# 6 7 7
#
# so yeah diff filters = diff results

import cv2
import numpy as np

def smoothing_box(image: np.ndarray) -> np.ndarray:
    """
    box blur → average filter
    """
    return cv2.blur(image, (15, 15))

def smoothing(image: np.ndarray) -> np.ndarray:
    """
    gaussian blur → weighted blur with center more important
    """
    return cv2.GaussianBlur(image, (15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT)

def smoothing_median(image: np.ndarray) -> np.ndarray:
    """
    median blur → takes the median of window, good for salt pepper noise
    """
    return cv2.medianBlur(image, 15)



