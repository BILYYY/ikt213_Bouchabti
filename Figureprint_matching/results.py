import numpy as np


class MatchResult:

    def __init__(self,
                 method_name: str,
                 num_matches: int,
                 num_kp1: int,
                 num_kp2: int,
                 processing_time: float,
                 match_image: np.ndarray = None):
        self.method_name = method_name
        self.num_matches = num_matches
        self.num_kp1 = num_kp1
        self.num_kp2 = num_kp2
        self.processing_time = processing_time
        self.match_image = match_image

    def __str__(self) -> str:
        return (
            f"{self.method_name}:\n"
            f"  Keypoints: {self.num_kp1} vs {self.num_kp2}\n"
            f"  Good Matches: {self.num_matches}\n"
            f"  Time: {self.processing_time:.4f}s"
        )

    def get_summary(self) -> dict:
        return {
            'method': self.method_name,
            'matches': self.num_matches,
            'keypoints_img1': self.num_kp1,
            'keypoints_img2': self.num_kp2,
            'time_seconds': self.processing_time
        }