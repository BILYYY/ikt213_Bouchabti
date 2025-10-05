from pipeline import MatchingPipeline


def main():

    pipeline = MatchingPipeline()

    img1 = "pictures/UiA front1.png"
    img2 = "pictures/UiA front3.jpg"

    # Run comparison
    orb_result, sift_result = pipeline.compare_methods(img1, img2, save_figure=True)

    # Print recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)

    # Check if matching was successful
    if orb_result.num_matches > 10 and sift_result.num_matches > 10:
        print("Both methods successfully matched the images!")

        # Recommend based on speed vs accuracy tradeoff
        if orb_result.processing_time < sift_result.processing_time:
            print("Best choice: ORB+BF (faster)")
        else:
            print("Best choice: SIFT+FLANN (more accurate)")
    else:
        print("Low match count detected. Consider:")
        print("  - Adjusting ratio threshold")
        print("  - Increasing nfeatures")
        print("  - Using SIFT+FLANN for better accuracy")


if __name__ == "__main__":
    main()