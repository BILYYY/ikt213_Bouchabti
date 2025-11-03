import cv2
import numpy as np

def align_images_orb(image_to_align, reference_image, max_features=1500, good_match_precent=0.15):
    g1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        raise SystemExit(1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    k = max(8, int(len(matches) * good_match_precent))
    matches = matches[:k]
    if len(matches) < 8:
        raise SystemExit(1)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None or mask is None or mask.sum() < 6:
        raise SystemExit(1)

    h, w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, H, (w, h))

    inliers = [m for m, inl in zip(matches, mask.ravel().tolist()) if inl]
    non_inliers = [m for m, inl in zip(matches, mask.ravel().tolist()) if not inl]
    extra = non_inliers[:min(20, len(non_inliers))]
    vis = inliers + extra

    matches_img = cv2.drawMatches(
        image_to_align, kp1, reference_image, kp2,
        vis, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return aligned, matches_img

if __name__ == "__main__":
    img1 = cv2.imread("align_this.jpg")
    img2 = cv2.imread("reference_img.png")
    if img1 is None or img2 is None:
        raise SystemExit(1)
    aligned, matches = align_images_orb(img1, img2, 1500, 0.15)
    cv2.imwrite("aligned.png", aligned)
    cv2.imwrite("matches.png", matches)
