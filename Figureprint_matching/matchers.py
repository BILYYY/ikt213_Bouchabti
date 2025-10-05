import cv2
import numpy as np


class FeatureMatcher:

    def __init__(self, ratio_threshold: float = 0.7):
        self.ratio_threshold = ratio_threshold
        self.matcher = None

    def match(self, des1: np.ndarray, des2: np.ndarray) -> list:
        if des1 is None or des2 is None:
            return []

        # Find 2 best matches for each descriptor
        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # m.distance < 0.7 * n.distance → reliable match
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches


class BFMatcher(FeatureMatcher):

    def __init__(self, ratio_threshold: float = 0.7):
        super().__init__(ratio_threshold)
        # NORM_HAMMING for binary descriptors (counts bit differences)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.name = "BF"


class FLANNMatcher(FeatureMatcher):

    def __init__(self, ratio_threshold: float = 0.7):
        super().__init__(ratio_threshold)
        # KD-tree algorithm for fast approximate search
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.name = "FLANN"