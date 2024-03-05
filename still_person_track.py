#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import os

from round_utils.dataload import LoadVideo
from round_utils.draw import draw_person
from round_utils.general import check_img_size


from person_detect import PersonDectect
from deep_reid import DeepReid, get_reid

from sklearn.metrics.pairwise import cosine_similarity


class StillPersonTracker:
    def __init__(self, still_threshold=1, still_frames=3):
        self.pos = [[], []]
        self.still_count = [0, 0]
        self.still_threshold = still_threshold ** 2  # 预先计算阈值的平方
        self.still_frames = still_frames

    def reset(self):
        self.pos = [[], []]
        self.still_count = [0, 0]

    def update(self, target_id, new_pos, index_frame, res):
        if len(self.pos[target_id]) == 0:
            self.pos[target_id] = [new_pos[0], new_pos[1]]
        else:
            old_pos = self.pos[target_id]
            dis_square = (new_pos[0] - old_pos[0]) ** 2 + (new_pos[1] - old_pos[1]) ** 2  # 计算距离的平方
            if dis_square < self.still_threshold:
                self.still_count[target_id] += 1
                if self.still_count[target_id] >= self.still_frames:
                    if index_frame not in res:
                        res.append(index_frame)
            else:
                self.still_count[target_id] = 0
            self.pos[target_id] = [new_pos[0], new_pos[1]]


class StillPersonDetect:
    def __init__(self, args):
        self.args = args

        # Person_detect行人检测类
        self.person_detect = PersonDectect(self.args)

        # deepreid 类
        self.deepreid = DeepReid(args)
        self.imgsz = check_img_size(args.img_size, s=32)  # check img_size

        self.fourcc = "mp4v"  # output video codec
        self.vid_writer = None

        self.tracker = StillPersonTracker(args.still_threshold, args.still_frames)  # 添加still_frames参数
        self.cut_frame = []
        self.generate_query_feat_and_name(args.query_path)


    def load_video(self, path, start, end):
        self.dataset = LoadVideo(path, img_size=self.imgsz, start=start, end=end)

    def update_by_singal_img(self, idx_frame, img, ori_img):
        # yolo detection
        bbox_xywh, xy = self.person_detect.detect(img, ori_img)

        # print("Processing img {}".format(idx_frame))

        if len(bbox_xywh) == 0:
            return
        
        features = self.deepreid.get_features(bbox_xywh, ori_img)

        # self.query_feat: 每个运动员取三个image，所以一共是6*512
        # person_cossim: yolo识别出的target分别与self.query_feat计算相似度，所以维度为： yolo识别出的目标数*6
        # max_idx： 在每个yolo识别出的target上选出6个中相似度最大的一个，即选出该目标和6个中的哪个更相似
        # max_idx[maximum < 0.6] = -1 意味着：不会对相似度不足0.6的目标做出操作

        person_cossim = cosine_similarity(features, self.query_feat)
        max_idx = np.argmax(person_cossim, axis=1)
        maximum = np.max(person_cossim, axis=1)
        max_idx[maximum < 0.6] = -1
        reid_results = max_idx

        draw_person(ori_img, xy, reid_results, self.names)

        # 遍历识别得到的结果，依次根据每个reid结果去update
        for index, index_on_name in enumerate(reid_results):
            if index_on_name == -1:
                continue
            else:
                x_c, y_c = bbox_xywh[index][:2]
                target_id = -1
                if index_on_name in [0, 1, 2]:
                    target_id = 0
                else:
                    target_id = 1

                self.tracker.update(target_id, [x_c, y_c], idx_frame, self.cut_frame)

    def detect(self):
        idx_frame = 0
        for img, ori_img, vid_cap in self.dataset:
            idx_frame += 1
            self.update_by_singal_img(idx_frame, img, ori_img, vid_cap)

    def generate_query_feat_and_name(self, query_path):
        path = str(Path(query_path))  # os-agnostic
        path = os.path.abspath(path)  # absolute path
        names = []
        extrator = self.deepreid.get_extractor()
        embs = np.ones((1, 512), dtype=int)
        # 从query_path/palyer1,query_path/player2中读取图片,person_name为player1,player2
        for person_name in ["player1", "player2"]:
            for image_name in os.listdir(os.path.join(path, person_name)):
                img = cv2.imread(os.path.join(path, person_name, image_name))
                feat = extrator.run_on_image(img)
                pytorch_output = feat.numpy()
                embs = np.concatenate((pytorch_output, embs), axis=0)
                names.append(person_name)
        names = names[::-1]
        names.append("None")
        self.query_feat = embs[:-1, :]
        self.names = names
        print("names: ", self.names)


def parse_args():
    parser = argparse.ArgumentParser("For yolo_reid")
    parser.add_argument(
        "--video_path",
        default="/home/user/tongchonggang/Yolov5-Deepsort-Fastreid/roundCut5.mp4",
        type=str,
    )

    parser.add_argument(
        "--device", default="cuda:0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    # yolov5
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="./weights/yolov5s.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--img-size", type=int, default=1088, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--classes",
        default=[0],
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")

    # fast_reid config
    parser.add_argument(
        "--query-path",
        default="/home/user/tongchonggang/Yolov5-Deepsort-Fastreid/fast_reid/query",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="kd-r34-r101_ibn/config.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="If use multiprocess for feature extraction.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "kd-r34-r101_ibn/model_final.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import time

    game_cutter = StillPersonDetect(args)
    game_cutter.load_video(args.video_path, 0, 600)
    game_cutter.generate_query_feat_and_name(args.query_path)
    t1 = time.time()
    with torch.no_grad():
        game_cutter.detect()
    print("cut res: {}".format(game_cutter.cut_frame))
    print("time is: ", time.time() - t1)
# [1, 214, 410]
# [84, 85, 87, 88, 89, 97, 102, 107, 112, 114, 117, 127, 132, 147, 152, 157, 167, 172, 177, 187, 192, 197, 207, 219, 224, 229, 234, 239, 244, 249, 254, 259, 264, 269, 279, 284, 285, 289, 294, 299, 304, 309, 314, 319, 324, 329, 334, 339, 344, 349, 354, 359, 369, 374, 379, 384, 389, 394, 399, 404, 409, 468, 478, 513, 558, 563, 568, 573, 578, 583, 588, 593, 598]
