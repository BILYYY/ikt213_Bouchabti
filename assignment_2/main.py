from src.shared.image_io import load_image, save_image
from src.task_padding import padding
from src.task_cropping import crop
from src.task_resize import resize
from src.task_copy import copy_lib
from src.task_grayscale import grayscale
from src.task_hsv import hsv
from src.task_hue_shift import hue_shifted
from src.task_smoothing import smoothing
from src.task_rotation import rotation
import numpy as np


INPUT_FILE = "lena-2.png"

OUTPUT_FILE_PAD = "lena_pad_reflect_100.png"
OUTPUT_FILE_CROP = "lena_cropped.png"
OUTPUT_FILE_RESIZE = "lena_200x200.png"
OUTPUT_FILE_COPY = "lena_copy.png"
OUTPUT_FILE_GRAY = "lena_gray.png"
OUTPUT_FILE_HSV = "lena_hsv.png"
OUTPUT_FILE_HUE = "lena_hue_shift_+50.png"
OUTPUT_FILE_BLUR = "lena_blur_15x15.png"
OUTPUT_FILE_ROT90 = "lena_rot90.png"
OUTPUT_FILE_ROT180 = "lena_rot180.png"

def run_task1():
    img = load_image(INPUT_FILE)
    save_image(padding(img, border_width=100), OUTPUT_FILE_PAD)
    print("✅ Padded image saved:", OUTPUT_FILE_PAD)

def run_task2():
    img = load_image(INPUT_FILE)
    h, w, _ = img.shape
    x0, y0 = 80, 80
    x1, y1 = w - 130, h - 130
    save_image(crop(img, x0, x1, y0, y1), OUTPUT_FILE_CROP)
    print("✅ Cropped image saved:", OUTPUT_FILE_CROP)

def run_task3():
    img = load_image(INPUT_FILE)
    save_image(resize(img, width=200, height=200), OUTPUT_FILE_RESIZE)
    print("✅ Resized image saved:", OUTPUT_FILE_RESIZE)

def run_task4():
    img = load_image(INPUT_FILE)
    save_image(copy_lib(img), OUTPUT_FILE_COPY)
    print("✅ Copied image saved:", OUTPUT_FILE_COPY)

def run_task5():
    img = load_image(INPUT_FILE)
    save_image(grayscale(img), OUTPUT_FILE_GRAY)
    print("✅ Grayscale image saved:", OUTPUT_FILE_GRAY)

def run_task6():
    img = load_image(INPUT_FILE)
    save_image(hsv(img), OUTPUT_FILE_HSV)
    print("✅ HSV image saved:", OUTPUT_FILE_HSV)

def run_task7():
    img = load_image(INPUT_FILE)
    empty = np.zeros_like(img)
    save_image(hue_shifted(img, empty, hue=50), OUTPUT_FILE_HUE)
    print("✅ Hue-shifted image saved:", OUTPUT_FILE_HUE)

def run_task8():
    img = load_image(INPUT_FILE)
    save_image(smoothing(img), OUTPUT_FILE_BLUR)
    print("✅ Blurred image saved:", OUTPUT_FILE_BLUR)

def run_task9():
    img = load_image(INPUT_FILE)
    save_image(rotation(img, 90), OUTPUT_FILE_ROT90)
    save_image(rotation(img, 180), OUTPUT_FILE_ROT180)
    print("✅ Rotated images saved:", OUTPUT_FILE_ROT90, "and", OUTPUT_FILE_ROT180)

if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
    run_task4()
    run_task5()
    run_task6()
    run_task7()
    run_task8()
    run_task9()
