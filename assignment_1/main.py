import cv2
import os

def print_image_information(image):
    masurment = image.shape;
    h = masurment[0];
    w = masurment[1];
    c = masurment[2];
    print(f"Height: {h}\nWidth: {w}\nChannels: {c}\nSize (number of values in array): {image.size}\nData type: {image.dtype}");


def save_camera_info():
    cap = cv2.VideoCapture(0);
    fps = cap.get(cv2.CAP_PROP_FPS);
    if fps == 0:
        fps = 30;

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);

    save_dir = "solutions";
    os.makedirs(save_dir, exist_ok=True);

    filepath = os.path.join(save_dir, "camera_outputs.txt");
    with open(filepath, "w") as f:
        f.write(f"fps: {int(fps)}\n");
        f.write(f"height: {int(height)}\n");
        f.write(f"width: {int(width)}\n");

    print(f"Camera information saved at {filepath}");

    cap.release();

def main():
    image = cv2.imread("lena-1.png");
    print_image_information(image);

    save_camera_info();

if __name__ == "__main__":
    main();
