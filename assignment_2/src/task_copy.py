import numpy as np

def copy_lib(image: np.ndarray) -> np.ndarray:

    return image.copy()  # creates a new array with the same data
def copy_m(image: np.ndarray) -> np.ndarray:

    h, w, c = image.shape

    # Create an empty black image with the same size
    emptyPictureArray = np.zeros((h, w, c), dtype=np.uint8)

    # Loop over rows and columns, copy each pixel
    for i in range(h):
        for j in range(w):
            emptyPictureArray[i, j] = image[i, j]  # copy all channels at once

    return emptyPictureArray
