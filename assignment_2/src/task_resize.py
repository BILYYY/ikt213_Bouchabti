import cv2
import numpy as np

def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize with OpenCV ().
    - image : input ndarray (H×W×C)
    - width : target width
    - height: target height

    cv2.resize expects dsize = (new_width, new_height).
    We use INTER_LINEAR for smooth results (recommended default).
    """
    # Call OpenCV's resize
    out = cv2.resize(
        image,                    # source image
        dsize=(width, height),    # (W, H) in OpenCV
        interpolation=cv2.INTER_LINEAR
    )
    return out
def manual_resize_nn(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Mapping:
      src_i = floor( i_out * (H / height) )
      src_j = floor( j_out * (W / width) )
    """
    H, W, C = image.shape

    # Allocate output (height × width × C)
    out = np.empty((height, width, C), dtype=np.uint8)

    # Example (tiny) mapping idea:
    # Original 2×3:
    #  A B C
    #  D E F
    # Target 3×4:
    #  [A/A/B/B]
    #  [A/A/B/B]  (nearest neighbor duplicates nearest source pixels)
    #  [D/D/E/E]

    for i_out in range(height):          # loop over output rows
        # float source row (where this output row samples from)
        src_i_f = i_out * (H / height)
        # nearest integer index in source image
        src_i = int(np.floor(src_i_f))
        if src_i >= H:                   # clamp for safety
            src_i = H - 1

        for j_out in range(width):       # loop over output cols
            src_j_f = j_out * (W / width)
            src_j = int(np.floor(src_j_f))
            if src_j >= W:
                src_j = W - 1

            # copy pixel
            out[i_out, j_out] = image[src_i, src_j]

    return out


# === Libraries & Functions Used ===
#
# from cv2 (OpenCV):
# - cv2.resize(src, dsize=(width, height), interpolation)
#       Inputs:
#           src         : input image (ndarray, H×W×C)
#           dsize       : (new_width, new_height)
#           interpolation:
#               * cv2.INTER_LINEAR      → good default for shrink/enlarge (bilinear)
#               * cv2.INTER_NEAREST     → nearest-neighbor (blocky but simple)
#       Output:
#           resized image (ndarray)
#
# from numpy (imported as np):
# - np.ndarray         : image container (H×W×C)
# - np.linspace(a,b,N) : build N evenly spaced values from a to b (inclusive/exclusive note below)
# - np.floor / astype  : map float coords to integer indices
# - np.empty(...)      : allocate array without init
# - Indexing           : arr[rows, cols] → pick pixels
#
# === Functions defined in this file ===
#
# - resize(image: np.ndarray, width: int, height: int) -> np.ndarray
#       Purpose: Resize using OpenCV (bilinear). **This is the one to call in main.py**
#       Input:
#           image  → original (H×W×C)
#           width  → target width
#           height → target height
#       Output:
#           (height × width × C) image, smooth interpolation
#
# - resize_numpy_nn(image: np.ndarray, width: int, height: int) -> np.ndarray
#       Purpose: Resize without OpenCV (nearest-neighbor) using NumPy math (few lines, no loops)
#       Output:
#           (height × width × C) image, blocky look (nearest neighbor)
#
# - manual_resize_nn(image: np.ndarray, width: int, height: int) -> np.ndarray
#       Purpose: Fully manual nearest-neighbor with explicit for-loops (learn the mapping)
#       Output:
#           (height × width × C) image (same result as nearest-neighbor)
#
# === Notes on nearest-neighbor mapping ===
# For an output pixel at (i_out, j_out):
#   i_out in [0, height-1], j_out in [0, width-1]
# Map back to source coordinates:
#   src_i = floor( i_out * (H / height) )
#   src_j = floor( j_out * (W / width) )
# Then copy: out[i_out, j_out] = in[src_i, src_j]
# (Clamp to valid range if needed: src_i in [0, H-1], src_j in [0, W-1])
