import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from extractors import ORBExtractor, SIFTExtractor
from matchers import BFMatcher, FLANNMatcher
from image_matcher import ImageMatcher


class MatchingPipeline:
    """Runs both matching methods and compares results"""

    def __init__(self):
        # Create ORB+BF pipeline
        self.orb_bf = ImageMatcher(
            extractor=ORBExtractor(nfeatures=1000),
            matcher=BFMatcher(ratio_threshold=0.7)
        )

        # Create SIFT+FLANN pipeline
        self.sift_flann = ImageMatcher(
            extractor=SIFTExtractor(nfeatures=1000),
            matcher=FLANNMatcher(ratio_threshold=0.7)
        )

    def compare_methods(self, img1_path: str, img2_path: str,
                        save_figure: bool = True, save_results: bool = True):
        """
        Run both methods on same image pair and compare
        Returns tuple of (orb_result, sift_result)
        """
        print(f"\n{'=' * 70}")
        print(f"Comparing: {Path(img1_path).name} vs {Path(img2_path).name}")
        print(f"{'=' * 70}\n")

        # Run ORB+BF
        orb_result = self.orb_bf.match_images(img1_path, img2_path)
        print(orb_result)

        # Run SIFT+FLANN
        sift_result = self.sift_flann.match_images(img1_path, img2_path)
        print(f"\n{sift_result}")

        # Print analysis
        self._print_analysis(orb_result, sift_result)

        # Visualize results
        if save_figure:
            self._visualize_results(orb_result, sift_result, img1_path, img2_path)

        # Save to text file
        if save_results:
            self._save_to_file(orb_result, sift_result, img1_path, img2_path)

        return orb_result, sift_result

    def _print_analysis(self, orb_result, sift_result):
        """Compare and print performance analysis"""
        print(f"\n{'=' * 70}")
        print("ANALYSIS:")
        print(f"{'=' * 70}")

        # Speed comparison
        if orb_result.processing_time < sift_result.processing_time:
            speed_diff = ((sift_result.processing_time - orb_result.processing_time)
                          / sift_result.processing_time * 100)
            print(f"Speed: ORB+BF is {speed_diff:.1f}% faster")
        else:
            speed_diff = ((orb_result.processing_time - sift_result.processing_time)
                          / orb_result.processing_time * 100)
            print(f"Speed: SIFT+FLANN is {speed_diff:.1f}% faster")

        # Match comparison
        match_diff = abs(orb_result.num_matches - sift_result.num_matches)
        winner = "ORB+BF" if orb_result.num_matches > sift_result.num_matches else "SIFT+FLANN"
        print(f"Matches: {winner} found {match_diff} more matches")

        # Characteristics
        print(f"Resources: ORB+BF uses binary descriptors (efficient)")
        print(f"Quality: SIFT+FLANN uses floating-point (accurate)")

    def _visualize_results(self, orb_result, sift_result, img1_path, img2_path):
        """Create and save visualization figure"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # Plot ORB+BF results
        if orb_result.match_image is not None:
            axes[0].imshow(cv2.cvtColor(orb_result.match_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title(
                f"ORB+BF: {orb_result.num_matches} matches "
                f"in {orb_result.processing_time:.3f}s",
                fontsize=14, fontweight='bold'
            )
            axes[0].axis('off')

        # Plot SIFT+FLANN results
        if sift_result.match_image is not None:
            axes[1].imshow(cv2.cvtColor(sift_result.match_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(
                f"SIFT+FLANN: {sift_result.num_matches} matches "
                f"in {sift_result.processing_time:.3f}s",
                fontsize=14, fontweight='bold'
            )
            axes[1].axis('off')

        plt.tight_layout()

        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save figure to results folder
        output_name = results_dir / f"comparison_{Path(img1_path).stem}_vs_{Path(img2_path).stem}.png"
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"\nImage saved to: {output_name}")
        plt.show()
    def _save_to_file(self, orb_result, sift_result, img1_path, img2_path):
        """Save results to text file"""
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"matching_results_{timestamp}.txt"

        # Write results to file
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("IMAGE MATCHING RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image 1: {Path(img1_path).name}\n")
            f.write(f"Image 2: {Path(img2_path).name}\n\n")

            f.write("-" * 70 + "\n")
            f.write("ORB + BF MATCHER RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Keypoints (img1): {orb_result.num_kp1}\n")
            f.write(f"Keypoints (img2): {orb_result.num_kp2}\n")
            f.write(f"Good Matches: {orb_result.num_matches}\n")
            f.write(f"Processing Time: {orb_result.processing_time:.4f}s\n\n")

            f.write("-" * 70 + "\n")
            f.write("SIFT + FLANN MATCHER RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Keypoints (img1): {sift_result.num_kp1}\n")
            f.write(f"Keypoints (img2): {sift_result.num_kp2}\n")
            f.write(f"Good Matches: {sift_result.num_matches}\n")
            f.write(f"Processing Time: {sift_result.processing_time:.4f}s\n\n")

            # Comparison
            f.write("-" * 70 + "\n")
            f.write("COMPARISON:\n")
            f.write("-" * 70 + "\n")

            speed_diff = abs(orb_result.processing_time - sift_result.processing_time)
            faster = "ORB+BF" if orb_result.processing_time < sift_result.processing_time else "SIFT+FLANN"
            f.write(f"Speed Winner: {faster} (by {speed_diff:.4f}s)\n")

            match_winner = "ORB+BF" if orb_result.num_matches > sift_result.num_matches else "SIFT+FLANN"
            match_diff = abs(orb_result.num_matches - sift_result.num_matches)
            f.write(f"More Matches: {match_winner} (+{match_diff} matches)\n\n")

            # Recommendation
            f.write("-" * 70 + "\n")
            f.write("RECOMMENDATION:\n")
            f.write("-" * 70 + "\n")
            if orb_result.num_matches > 10 and sift_result.num_matches > 10:
                f.write("Both methods successfully matched the images.\n")
                f.write(f"Best choice: {faster} (faster)\n")
            else:
                f.write("Low match count. Consider adjusting parameters.\n")

        print(f"Results saved to: {filename}")