import cv2
import numpy as np


def harris_corner_detection(reference_image):
    if len(reference_image.shape) == 3:
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = reference_image.copy()

    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    dst = cv2.dilate(dst, None)

    img_with_corners = reference_image.copy()

    img_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img_with_corners


if __name__ == "__main__":
    print("Running Harris Corner Detection...")

    # Load reference image
    reference_img = cv2.imread('reference_img.png')

    if reference_img is None:
        print("Error: Could not load reference_img.png")
        exit(1)

    # Perform Harris corner detection
    harris_result = harris_corner_detection(reference_img)

    # Save result
    cv2.imwrite('harris.png', harris_result)
    print("Saved: harris.png")
    print("Harris corner detection completed!")