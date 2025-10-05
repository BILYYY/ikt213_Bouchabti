import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """
    Handles image loading and preprocessing operations
    Used to prepare images before feature extraction
    """

    @staticmethod
    def load_and_preprocess(image_path: str) -> np.ndarray:
        """
        Load image from disk and apply Otsu's binarization

        Process:
        1. Load image in grayscale mode
        2. Apply automatic threshold using Otsu's method
        3. Invert binary image (ridges white, background black)

        Args:
            image_path: Path to image file (str or Path object)

        Returns:
            Binary image as numpy array (height × width, uint8)
            Values: 0 (black) or 255 (white)

        Raises:
            ValueError: If image cannot be loaded from path
        """
        # === Step 1: Load image in grayscale ===
        # cv2.IMREAD_GRAYSCALE → loads image as single channel (no colors)
        # Result: 2D array of pixel intensities (0-255)
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # === Step 2: Validate image loaded successfully ===
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        """
        === Step 3: Apply Otsu's binarization ===
        cv2.threshold() converts grayscale to pure black/white
        Parameters breakdown:
          - img: input grayscale image
          - 0: threshold value (ignored when using OTSU)
          - 255: max value for pixels above threshold
          - cv2.THRESH_BINARY_INV: inverse binary (below threshold = 255)
          - cv2.THRESH_OTSU: automatically finds optimal threshold

        Returns:
          - threshold_value: the calculated threshold (we ignore with _)
          - img_binary: the binarized image

        THRESH_BINARY_INV explanation:
          Normal: pixel < threshold → 0, pixel >= threshold → 255
          Inverted: pixel < threshold → 255, pixel >= threshold → 0
          For fingerprints: ridges become white (255), valleys black (0)
        """
        _, img_binary = cv2.threshold(
            img,
            0,  # threshold (auto-calculated)
            255,  # max value
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  # method flags
        )

        return img_binary
"""
=== Libraries Used ===

cv2 (OpenCV):
- cv2.imread(path, flag)
      Loads an image from file
      Inputs:
          path: file path as string
          flag: color mode (cv2.IMREAD_GRAYSCALE = single channel)
      Output: numpy array or None if failed

- cv2.IMREAD_GRAYSCALE
      Flag for imread() to load image in grayscale mode
      Result: 2D array (height × width) instead of 3D (height × width × 3)

- cv2.threshold(src, thresh, maxval, type)
      Applies fixed-level threshold to image
      Inputs:
          src: grayscale image
          thresh: threshold value (0-255)
          maxval: value assigned to pixels above threshold
          type: thresholding method flags
      Output: (threshold_value, binary_image)

- cv2.THRESH_BINARY_INV
      Inverted binary threshold flag
      Below threshold → maxval (255)
      Above threshold → 0

- cv2.THRESH_OTSU
      Automatic threshold calculation flag
      Uses Otsu's method to find optimal threshold value
      Analyzes image histogram to minimize intra-class variance

numpy (as np):
- np.ndarray
      N-dimensional array type
      For images: typically 2D (grayscale) or 3D (color)

pathlib:
- Path
      Object-oriented file path handling
      Allows cross-platform path operations

"""