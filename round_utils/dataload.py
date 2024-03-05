from pathlib import Path
import os
import glob
import cv2
import numpy as np

img_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
vid_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]


class LoadVideo:  # for inference
    def __init__(self, path, img_size=640, start=None, end=None):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        assert (
            os.path.splitext(p)[-1].lower() in vid_formats
        ), "video formats error. Supported formats are: \nvideos: %s" % (vid_formats)

        self.img_size = img_size

        self.start = start
        self.end = end

        self.new_video(p)

    def __iter__(self):
        self.frame = 0
        return self

    def __next__(self):
        if self.frame == self.nframes:
            raise StopIteration

        ret_val, img0 = self.cap.read()
        if not ret_val:
            self.cap.release()
            raise StopIteration

        self.frame += 1

        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # return img, img0, self.cap
        # return img, img0, cur_frame_index, self.cap
        return img, img0, self.cap, self.start + self.frame - 1

    def new_video(self, path):
        # self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        self.nframes = self.end - self.start

    def __len__(self):
        return self.nframes  # number of files


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)
