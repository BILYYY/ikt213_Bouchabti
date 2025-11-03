import cv2

def find_and_draw_matches_sift(image1, image2, max_features=10, good_match_percent=0.7):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    print(f"Found {len(keypoints1)} keypoints in image 1")
    print(f"Found {len(keypoints2)} keypoints in image 2")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches after ratio test")

    good_matches = sorted(good_matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(good_matches) * good_match_percent)
    good_matches = good_matches[:num_good_matches]

    print(f"Keeping top {len(good_matches)} matches ({good_match_percent * 100}%)")

    matches_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matches_image, good_matches


def find_and_draw_matches_orb(image1, image2, max_features=1500, good_match_percent=0.15):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    print(f"Found {len(keypoints1)} keypoints in image 1")
    print(f"Found {len(keypoints2)} keypoints in image 2")

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    print(f"Found {len(matches)} initial matches")

    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    print(f"Keeping top {len(matches)} matches ({good_match_percent * 100}%)")

    matches_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matches_image, matches


if __name__ == "__main__":
    print("Running Feature Matching...")

    # Load images
    image1 = cv2.imread('align_this.jpg')
    image2 = cv2.imread('reference_img.png')

    if image1 is None:
        print("Error: Could not load align_this.jpg")
        exit(1)
    if image2 is None:
        print("Error: Could not load reference_img.png")
        exit(1)

    method = 'ORB'

    if method == 'SIFT':
        matches_img, matches_list = find_and_draw_matches_sift(image1, image2,
                                                               max_features=4000,
                                                               good_match_percent=0.15)
    else:
        matches_img, matches_list = find_and_draw_matches_orb(image1, image2,
                                                              max_features=5000,
                                                              good_match_percent=0.20)

    cv2.imwrite('matches.png', matches_img)