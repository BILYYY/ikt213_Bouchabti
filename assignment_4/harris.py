import cv2
import numpy as np

def harris_corner_detection(reference_image):
    if len(reference_image.shape) == 3:
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = reference_image.copy()
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    out = reference_image.copy()
    out[dst > 0.01 * dst.max()] = [0, 0, 255]
    return out

if __name__ == "__main__":
    ref = cv2.imread("reference_img.png")
    if ref is None:
        raise SystemExit(1)
    harris_img = harris_corner_detection(ref)
    cv2.imwrite("harris.png", harris_img)
