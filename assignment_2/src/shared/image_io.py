import os
import cv2
import numpy as np

# chemins vers les dossiers
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")

def load_image(filename: str) -> np.ndarray:
    """Charge une image depuis data/"""
    path = os.path.join(DATA_DIR, filename)
    img = cv2.imread(path)   # image lue en BGR
    return img

def save_image(img, filename: str) -> str:
    """Sauvegarde une image dans outputs/"""
    out_path = os.path.join(OUTPUTS_DIR, filename)
    cv2.imwrite(out_path, img)
    return out_path
