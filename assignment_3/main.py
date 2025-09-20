import cv2
import numpy as np
import os

def sobel_edge_detection(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (3, 3), sigmaX=0)
    s = cv2.Sobel(b, cv2.CV_64F, dx=1, dy=1, ksize=1)
    s = np.absolute(s)
    s = np.uint8(s)
    cv2.imwrite('output/sobel_edge_detection.png', s)
    return s

def canny_edge_detection(img, t1, t2):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (3, 3), sigmaX=0)
    c = cv2.Canny(b, t1, t2)
    cv2.imwrite('output/canny_edge_detection.png', c)
    return c

def template_match(img, temp):
    ig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    h, w = tg.shape
    r = cv2.matchTemplate(ig, tg, cv2.TM_CCOEFF_NORMED)
    t = 0.9
    l = np.where(r >= t)
    m = img.copy()
    for pt in zip(*l[::-1]):
        cv2.rectangle(m, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite('output/template_match_result.png', m)
    return m

def resize(img, sf, ud):
    ci = img.copy()
    if ud.lower() == "up":
        for i in range(sf):
            ci = cv2.pyrUp(ci)
        f = f'output/resized_up_scale_{sf}.png'
    elif ud.lower() == "down":
        for i in range(sf):
            ci = cv2.pyrDown(ci)
        f = f'output/resized_down_scale_{sf}.png'
    cv2.imwrite(f, ci)
    return ci

def main():
    os.makedirs('output', exist_ok=True)
    li = cv2.imread('lambo.png')
    sobel_edge_detection(li)
    canny_edge_detection(li, 50, 50)
    si = cv2.imread('shapes-1.png')
    ti = cv2.imread('shapes_template.jpg')
    template_match(si, ti)
    resize(li, 2, "up")
    resize(li, 2, "down")

if __name__ == "__main__":
    main()