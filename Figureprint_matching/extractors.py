import cv2
import numpy as np


class FeatureExtractor:
    """Base class for feature extraction"""

    def __init__(self, nfeatures: int = 1000):
        self.nfeatures = nfeatures
        self.detector = None

    def extract(self, image: np.ndarray) -> tuple:
        """Extract keypoints and descriptors from image"""
        if self.detector is None:
            raise NotImplementedError("Detector not initialized")

        # Returns (keypoints, descriptors)
        return self.detector.detectAndCompute(image, None)


class ORBExtractor(FeatureExtractor):
    """ORB feature extractor - fast, binary descriptors"""

    def __init__(self, nfeatures: int = 1000):
        super().__init__(nfeatures)
        self.detector = cv2.ORB_create(nfeatures=nfeatures)
        self.name = "ORB"


class SIFTExtractor(FeatureExtractor):
    """SIFT feature extractor - accurate, floating-point descriptors"""

    def __init__(self, nfeatures: int = 1000):
        super().__init__(nfeatures)
        self.detector = cv2.SIFT_create(nfeatures=nfeatures)
        self.name = "SIFT"