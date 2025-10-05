import cv2
import time
from preprocessor import ImagePreprocessor
from results import MatchResult


class ImageMatcher:
    """Combines feature extraction and matching into single pipeline"""

    def __init__(self, extractor, matcher):
        self.extractor = extractor
        self.matcher = matcher
        self.preprocessor = ImagePreprocessor()

    def match_images(self, img1_path: str, img2_path: str,
                     draw_matches: bool = True) -> MatchResult:
        """
        Complete matching pipeline for two images
        Returns MatchResult with all data
        """
        # Start timing
        start_time = time.time()

        # Load and binarize images
        img1 = self.preprocessor.load_and_preprocess(img1_path)
        img2 = self.preprocessor.load_and_preprocess(img2_path)

        # Extract features (keypoints + descriptors)
        kp1, des1 = self.extractor.extract(img1)
        kp2, des2 = self.extractor.extract(img2)

        # Match descriptors between images
        good_matches = self.matcher.match(des1, des2)

        # Draw visualization if requested
        match_img = None
        if draw_matches and len(good_matches) > 0:
            # Creates side-by-side image with lines connecting matches
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        # Calculate elapsed time
        processing_time = time.time() - start_time

        # Build method name (e.g., "ORB+BF")
        method_name = f"{self.extractor.name}+{self.matcher.name}"

        # Package results
        return MatchResult(
            method_name=method_name,
            num_matches=len(good_matches),
            num_kp1=len(kp1) if kp1 else 0,
            num_kp2=len(kp2) if kp2 else 0,
            processing_time=processing_time,
            match_image=match_img
        )