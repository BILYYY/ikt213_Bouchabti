import cv2
import numpy as nu


def padding(image: nu.ndarray, border_width: int) -> nu.ndarray:
    """
    Ajoute une bordure miroir autour de l'image
    """
    padded = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT
    )
    return padded
import numpy as np

def zero_padding(image: np.ndarray, border_width: int) -> np.ndarray:
    h, w, c = image.shape   # original height, width, channels
    p = border_width

    new_h = h + 2*p
    new_w = w + 2*p

    padded_list = []  # start empty matrix

    # =========================
    # Example for h=2, w=3, p=1
    # Original image:
    #   A B C
    #   D E F
    #
    # New image size: 4x5
    #
    # Before filling (empty):
    # i\j   0   1   2   3   4
    #  0   ??? ??? ??? ??? ???
    #  1   ??? ??? ??? ??? ???
    #  2   ??? ??? ??? ??? ???
    #  3   ??? ??? ??? ??? ???
    # =========================

    for i in range(new_h):      # loop rows
        row = []

        for j in range(new_w):  # loop columns

            # --- TOP border ---
            if i < p:
                pixel = [0] * c  # black
                # Example: i=0, j=0..4
                # Row 0 becomes: [0][0][0][0][0]

            # --- BOTTOM border ---
            elif i >= h + p:
                pixel = [0] * c
                # Example: i=3 (last row), j=0..4
                # Row 3 becomes: [0][0][0][0][0]

            # --- LEFT border ---
            elif j < p:
                pixel = [0] * c
                # Example: i=1, j=0 → leftmost column is black

            # --- RIGHT border ---
            elif j >= w + p:
                pixel = [0] * c
                # Example: i=1, j=4 → rightmost column is black

            # --- CENTER region ---
            else:
                src_i = i - p   # map to original image row
                src_j = j - p   # map to original image col
                pixel = image[src_i, src_j].tolist()
                # Example: i=1, j=1 → src_i=0, src_j=0 → copy "A"
                # Example: i=1, j=2 → src_i=0, src_j=1 → copy "B"

            row.append(pixel)

        padded_list.append(row)

        # =========================
        # Debug: show matrix row by row
        # After i=0:
        # [0][0][0][0][0]
        #
        # After i=1:
        # [0][A][B][C][0]
        #
        # After i=2:
        # [0][D][E][F][0]
        #
        # After i=3:
        # [0][0][0][0][0]
        # =========================


    # Walkthrough
    # for the first 2 steps (h=2, w=3, p=1)
    #
    # (empty):
    #
    # ??? ??? ??? ??? ???
    # ??? ??? ??? ??? ???
    # ??? ??? ??? ??? ???
    # ??? ??? ??? ??? ???
    #
    #
    # After i = 0(top border row)
    #
    # [0][0][0][0][0]
    # ??? ??? ??? ??? ???
    # ??? ??? ??? ??? ???
    # ??? ??? ??? ??? ???
    #
    #
    # After i = 1(first row of image copied):
    #
    # [0][0][0][0][0]
    # [0][A][B][C][0]
    # ??? ??? ??? ??? ???
    # ??? ??? ??? ??? ???
    #
    #
    # After  i = 2(second row of image copied):
    #
    # [0][0][0][0][0]
    # [0][A][B][C][0]
    # [0][D][E][F][0]
    # ??? ??? ??? ??? ???

    padded = np.array(padded_list, dtype=np.uint8)
    return padded

# === Libraries & Functions Used ===
#
# from numpy (imported as np):
# - np.ndarray :
#       A data type representing an image in memory as a 3D array
#       Shape = (height, width, channels)
#       Example: (512, 512, 3) → 512 rows, 512 cols, 3 color channels (B,G,R)
#
# - np.zeros(shape, dtype) :
#       Creates a new array filled with zeros
#       Input: shape (tuple of dimensions), dtype (like np.uint8)
#       Output: ndarray of the given size, filled with 0
#       Example: np.zeros((5,5,3), np.uint8) → 5×5 black image with 3 channels
#
# from cv2 (OpenCV):
# - cv2.copyMakeBorder(src, top, bottom, left, right, borderType)
#       Adds a border around an image
#       Inputs:
#           src: input image (ndarray)
#           top/bottom/left/right: number of pixels to add on each side
#           borderType: method for filling the border
#       Output:
#           new image (ndarray) with larger size including the border
#
#   borderType used here:
#       * cv2.BORDER_REFLECT :
#           The border is filled by reflecting the pixels of the image like a mirror.
#           Example: [A B C] with p=2 → [B A A B C C B]
#
# === Functions defined in this file ===
#
# - padding(image: np.ndarray, border_width: int) -> np.ndarray
#       Purpose: Add a mirror border around the image
#       Input:
#           image → original ndarray (height × width × channels)
#           border_width → number of pixels to add on each side
#       Output:
#           new ndarray, bigger by +2*border_width in both dimensions,
#           with the edges mirrored outward
#
# - zero_padding(image: np.ndarray, border_width: int) -> np.ndarray
#       Purpose: Add a black border (all zeros) around the image
#       Input:
#           image → original ndarray
#           border_width → number of pixels for the border
#       Output:
#           new ndarray with +2*border_width in both height and width,
#           where the outer area is black and the center is the original image
