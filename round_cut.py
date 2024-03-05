#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import os
import json

from round_utils.dataload import LoadVideo
from round_utils.general import check_img_size
from round_utils.draw import draw_person


from person_detect import PersonDectect
from deep_reid import DeepReid

from sklearn.metrics.pairwise import cosine_similarity


class AthleteTracker:
    def __init__(self):
        self.positions = {}
        self.cross_by_id = -1

    def update(self, athlete_id, new_position, frame_index, table_center, cross_frames):
        old_position = self.positions.get(athlete_id, new_position)
        updated = False
        if self.is_cross_table(old_position, new_position, table_center):
            if self.cross_by_id == -1:
                cross_frames.append(frame_index)
                self.cross_by_id = athlete_id
                updated = True
            elif self.cross_by_id != athlete_id:
                self.cross_by_id = -1
            else:
                pass

        self.positions[athlete_id] = new_position
        return updated

    def is_cross_table(self, old_position, new_position, table_center):
        return (old_position[0] - table_center[0]) * (new_position[0] - table_center[0]) < 0


class RoundCut:
    def __init__(self, table_loc, args):
        self.args = args

        # Person_detect行人检测类
        self.person_detect = PersonDectect(self.args)

        # deepreid 类
        self.deepreid = DeepReid(args)
        self.imgsz = check_img_size(args.img_size, s=64)  # check img_size

        self.fourcc = "mp4v"  # output video codec
        self.vid_writer = None

        self.table_loc = table_loc

        self.tracker = AthleteTracker()
        self.cut_frame = []

        self.preview_path = args.preview_dir
        self.progress_path = args.progress_path
        self.progress_interval = args.progress_interval

        self.player1_num = args.player1_num
        self.player2_num = args.player2_num

        # 如果preview文件夹不存在则创建
        if not os.path.exists(self.preview_path):
            os.makedirs(self.preview_path)

        # 如果progress文件不存在则创建
        if not os.path.exists(self.progress_path):
            with open(self.progress_path, 'w') as file:
                json.dump({"progress": 0}, file)

    def load_video(self, path):
        video = cv2.VideoCapture(path)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        print("fps: ", self.fps)
        self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dataset = LoadVideo(
            path, img_size=self.imgsz, start=0, end=self.total_frames)

    def detect(self):
        for img, ori_img, vid_cap, idx_frame in self.dataset:
            if idx_frame not in self.table_loc:
                continue

            # yolo detection
            bbox_xywh, xy = self.person_detect.detect(img, ori_img)

            if idx_frame % 1000 == 0:
                print(f"正在处理 {idx_frame}, 剩余{self.dataset.end - idx_frame}帧")
                print(f"追踪结果: {self.cut_frame}")

            if len(bbox_xywh) == 0:
                continue

            features = self.deepreid.get_features(bbox_xywh, ori_img)

            person_cossim = cosine_similarity(features, self.query_feat)
            max_idx = np.argmax(person_cossim, axis=1)
            maximum = np.max(person_cossim, axis=1)
            max_idx[maximum < 0.6] = -1
            reid_results = max_idx

            updated = False
            candidates = {}
            for index, index_on_name in enumerate(reid_results):
                if index_on_name == -1:
                    continue
                else:
                    x_c, y_c = bbox_xywh[index][:2]
                    target_id = -1
                    # index_on_name in [0,self.player2_num-1]
                    if index_on_name in range(self.player2_num):
                        target_id = 1
                    else:
                        target_id = 0

                    # Add the target to the candidates dictionary
                    if target_id not in candidates or maximum[index] > candidates[target_id][1]:
                        candidates[target_id] = ([x_c, y_c], maximum[index])

            # Only update the tracker with the target that has the highest possibility
            if candidates:
                target_id, (position, _) = max(candidates.items(), key=lambda x: x[1][1])
                updated = self.tracker.update(target_id, position, idx_frame, self.table_loc[idx_frame], self.cut_frame)

            self.update_image_file(idx_frame, ori_img, xy, reid_results, updated)

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

    def update_image_file(self, frame_index, image, xy, reid_results, force=False):
        if frame_index % self.progress_interval == 0 or force:
            self.save_image(image, frame_index, xy, reid_results)
            self.save_progress(round(frame_index / self.total_frames * 100, 2))

        if frame_index >= self.total_frames:
            self.save_image(image, frame_index, xy, reid_results)
            self.save_progress(100)

    def save_image(self, image, frame_index, xy, reid_results):
        draw_person(image, xy, reid_results, self.names)

        # 保存图片
        cv2.imwrite(os.path.join(self.preview_path,
                    f"{frame_index}.jpg"), image)

    def save_progress(self, percentage):
        # 更新进度文件
        print("processing: ", percentage, "%")
        with open(self.progress_path, 'w') as file:
            json.dump({"progress": percentage}, file)

    def filter_values(self, lst=None):
        if lst is None:
            lst = self.cut_frame
        if not lst:
            return []
        res = [lst[0]]
        for i in range(1, len(lst)):
            if lst[i] - res[-1] >= self.args.filter_interval:
                res.append(lst[i])
        return res

    def get_segments(self, res):
        start = 0
        end = self.total_frames
        segments = []
        for i in res:
            if start < i <= end:
                segments.append([start, i])
                start = i
        if start < end:
            segments.append([start, end])

        for i in range(len(segments)):
            segments[i][0] = segments[i][0]/self.fps
            segments[i][1] = segments[i][1]/self.fps
        return segments


def parse_args():
    parser = argparse.ArgumentParser("For yolo_reid")
    parser.add_argument(
        "--video_path",
        default="game_cut_demo.mp4",
        type=str,
    )

    parser.add_argument(
        "--output_path",
        default="game_output/result.json",
        type=str,
    )

    parser.add_argument(
        "--query_path",
        default="fast_reid/query",
        type=str,
    )

    parser.add_argument(
        "--player2_num",
        default=13,
        type=int,
    )

    parser.add_argument(
        "--player1_num",
        default=12,
        type=int,
    )

    parser.add_argument(
        "--table_loc_path",
        default="table_loc.json",
        type=str,
    )

    parser.add_argument(
        "--filter_interval",
        default=100,
        type=int,
    )

    # preview image dir path
    parser.add_argument('--preview_dir', type=str,
                        default='app/useful_clip_analyse/datasets/results/preview', help='preview dir path')

    # progress file path
    parser.add_argument('--progress_path', type=str,
                        default='app/useful_clip_analyse/datasets/results/progress.json', help='progress file path')

    # progress save interval(frames)
    parser.add_argument('--progress_interval', type=int,
                        default=60, help='progress save interval(frames)')

    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    # yolov5
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
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")

    # fast_reid config
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


def filter_values(lst,interval=100):
    return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] - lst[i - 1] >= interval]

def get_segments(total_frames, res, fps, need_divide=True):
    start = 0
    end = total_frames
    segments = []
    for i in res:
        if start < i <= end:
            segments.append([start, i-1])
            start = i
    if start < end:
        segments.append([start, end])
    for i in range(len(segments)):
        segments[i][0] = need_divide and segments[i][0]/fps or segments[i][0]
        segments[i][1] = need_divide and segments[i][1]/fps or segments[i][1]
    return segments

if __name__ == "__main__":
    args = parse_args()

    # # Read json from args.table_loc_path
    with open(args.table_loc_path, 'r') as file:
        table_loc = json.load(file)
    
    # construct table_loc:{idx_frame: [table_center_x, table_center_y]}, and the
    # now table_loc is a dict which is {"idx_frame": {"table": [table_center_x, table_center_y, table_width, table_height]}}
    # just keep the float(table_center_x, table_center_y)
    table_loc = {int(k)-1: [float(v["table"][0]), float(v["table"][1])] for k, v in table_loc.items()}


    round_cutter = RoundCut(table_loc, args)
    round_cutter.load_video(args.video_path)
    round_cutter.generate_query_feat_and_name(args.query_path)
    round_cutter.detect()
    print("cut res: {}".format(round_cutter.cut_frame))
    
    # 过滤掉间隔小于100帧的
    filtered_res = filter_values(round_cutter.cut_frame, interval=100)
    # 生成片段
    print("filtered_res: ", filtered_res)

    # filtered_res = [3425, 6399, 9391, 13446]
    segments = get_segments(round_cutter.total_frames, filtered_res, round_cutter.fps, False)
    print("segments: ", segments)
    
    
    # # 将结果写入json文件
    import json
    with open(args.output_path, "w") as f:
        json.dump({"clips": segments}, f)
