import numpy as np

def crop(image: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """
    Crop the image using slicing.
    Inputs:
        image: original ndarray (height × width × channels)
        x0, x1: horizontal limits (columns)
        y0, y1: vertical limits (rows)
    """
    # Slicing in numpy → image[y_start:y_end, x_start:x_end]
    # Example: image[y0:y1, x0:x1]
    # - Keeps rows from y0 up to y1-1
    # - Keeps cols from x0 up to x1-1
    #
    # Example with a 3×4 image:
    # Original:
    # i\j   0   1   2   3
    #  0    A   B   C   D
    #  1    E   F   G   H
    #  2    I   J   K   L
    #
    # Crop with x0=0, x1=3, y0=0, y1=3 → result is:
    # A B C
    # E F G
    # I J K
    #
    cropped = image[y0:y1, x0:x1]
    return cropped

def m_crop(image: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:

    h, w, c = image.shape  # original dimensions

    new_h = y1 - y0        # new height after cropping
    new_w = x1 - x0        # new width after cropping

    cropped_list = []      # empty list to store rows

    # === Example small image ===
    # Original (4×5):
    # A B C D E
    # F G H I J
    # K L M N O
    # P Q R S T
    #
    # Crop with x0=1, x1=4, y0=1, y1=3 → keep only this box:
    # G H I
    # L M N
    # ============================

    # loop through the rows of the new cropped image
    for i in range(new_h):
        row = []
        for j in range(new_w):
            # map to the original image
            src_i = y0 + i   # original row index
            src_j = x0 + j   # original col index
            pixel = image[src_i, src_j].tolist()
            row.append(pixel)
        cropped_list.append(row)


    cropped = np.array(cropped_list, dtype=np.uint8)
    return cropped