import cv2
import numpy as np
import torch
import argparse
import bisect
import os
import json
from round_utils.dataload import LoadVideo
from round_utils.general import check_img_size

from paddleocr import PaddleOCR, draw_ocr

from still_person_track import StillPersonDetect


class ScoreTracker:
    def __init__(self):
        self.game_score = [0, 0]
        self.round_score = [0, 0]
        self.results = [{'game_score': [0, 0], 'round_score': [0, 0], 'end_frame': -1}]
        self.cut_frame=[]
        self.current_frame = 0

    def update(self, new_score, id_frame):
        self.current_frame = id_frame
        if self.is_valid_score(new_score):
            if self.is_score_unchanged(new_score):
                return
            
            if new_score == [0,0] and any(score >= 10 for score in self.round_score):
                # new round has begun
                winner = self.round_score.index(max(self.round_score))
                self.game_score[winner]+=1
                self.round_score = new_score
                self.cut_frame.append(id_frame)
                self.update_results()
                return

            if self.is_single_score_incremented(new_score):
                self.round_score=new_score
                self.update_results()
                self.cut_frame.append(id_frame)
                return
                
    def update_results(self):
        self.results[-1]['end_frame'] = self.current_frame - 1
        self.results.append({'game_score': self.game_score.copy(), 'round_score': self.round_score.copy(), 'end_frame': -1})

    def is_valid_score(self, score):
        return all(isinstance(i, int) for i in score)

    def is_score_unchanged(self, new_score):
        return new_score == self.round_score

    def is_single_score_incremented(self, new_score):
        total_increment = sum(new - old for new, old in zip(new_score, self.round_score))
        return 0 < total_increment <= 2



class ScoreCut:
    def __init__(self, args):
        self.args = args
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang="en", use_gpu=False, show_log=False
        )
        self.tracker = ScoreTracker()
        self.still_person_detector = StillPersonDetect(self.args)
        self.imgsz = check_img_size(args.img_size, s=64)

        self.final_res=[]

        self.preview_path = args.preview_dir
        self.progress_path = args.progress_path
        self.progress_interval = args.progress_interval

        # 如果preview文件夹不存在则创建
        if not os.path.exists(self.preview_path):
            os.makedirs(self.preview_path)

        # 如果progress文件不存在则创建
        if not os.path.exists(self.progress_path):
            with open(self.progress_path, 'w') as file:
                json.dump({"progress": 0}, file)
        
        # 将args中的x_start,x_end,y_start,y_end转化为int
        self.x_start = int(self.args.x_start)
        self.x_end = int(self.args.x_end)
        self.y_start = int(self.args.y_start)
        self.y_end = int(self.args.y_end)

    def load_video(self, path, start=0, end=-1):
        video = cv2.VideoCapture(path)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if end == -1:
            end = self.total_frames
            start = 0
        self.dataset = LoadVideo(
            path, self.imgsz, start, end)

    def detect(self):
        for img, ori_img, _ , idx_frame in self.dataset:
            ocr_res = self.ocr_process(idx_frame, ori_img)
            # self.still_person_detector.update_by_singal_img(
            #     idx_frame, img, ori_img
            # )

            # (x_start,y_start),(x_end,y_end),画一个矩形框
            cv2.rectangle(ori_img, (self.x_start, self.y_start), (self.x_end, self.y_end), (0, 0, 255), 2)
            self.update_progress(idx_frame,ori_img,ocr_res)
            

    def update_progress(self, idx_frame,ori_img,ocr_res):
        if idx_frame % self.progress_interval == 0 or idx_frame == self.total_frames:
            # 保存进度
            progress = round(idx_frame / self.total_frames * 100, 2)
            print("progress: {}%".format(progress))
            with open(self.progress_path, 'w') as file:
                json.dump({"progress": progress}, file)

            # 保存预览图,文件名为idx_frame_ocr_res[0]_ocr_res[1].png
            cv2.imwrite(
                os.path.join(
                    self.preview_path,
                    "{}_{}_{}.png".format(
                        idx_frame, ocr_res[0], ocr_res[1]
                    ),
                ),
                ori_img,
            )

    def ocr_process(self, idx_frame, ori_img):
        ocr_res = []
        for i in [0, 1]:
            y_length = int((self.y_end - self.y_start) / 2)
            y_start = self.y_start + i * y_length
            y_end = y_start + y_length
            ocr_img = ori_img[
                y_start:y_end,
                self.x_start : self.x_end,
            ]

            # 画出来
            # cv2.imwrite("round_output/ocr_{}.png".format(idx_frame), ori_img)
            # 将图像除了白色部分都变成黑色
            # ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)

            self.ocr_img_resize = self.resize_img(ocr_img)
            res = self.ocr.ocr(self.ocr_img_resize, True, True, False)

            print_ocr=[]

            if not res or res==[None]:
                ocr_res.append(0)
            else:
                result = res[0]
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                
                print_ocr.append(txts[0])

                # 如果字符串txt[0]可以转化为数字，则将ocr_res[i]设置为txt[0]，否则设置为0
                if txts[0].isdigit():
                    ocr_res.append(int(txts[0]))
                else:
                    ocr_res.append(0)

        print("ocr_res: {}, print_ocr: {}".format(ocr_res, print_ocr))

        if len(ocr_res)!=0:
            self.tracker.update(ocr_res, idx_frame)
        return ocr_res

    def resize_img(self, image):
        height, width = image.shape[0], image.shape[1]
        img_new = cv2.resize(image, (175, 175), interpolation=cv2.INTER_LINEAR)
        return img_new

    # 返回帧/fps的结果
    def get_ocr_res(self, divide_fps=True):
        if divide_fps:
            return [frame / self.fps for frame in self.tracker.cut_frame]
        else:
            return self.tracker.cut_frame

    def get_still_tracker_res(self, divide_fps=True):
        if divide_fps:
            return [frame / self.fps for frame in self.still_person_detector.cut_frame]
        else:
            return self.still_person_detector.cut_frame

    def lower_bound(self, lst, value):
        return bisect.bisect_left(lst, value)

    def get_final_res(self,ocr_res,reid_res):
        self.final_res.clear()
        for idx in ocr_res:
            index = self.lower_bound(reid_res, idx)
            if index == len(reid_res) or abs(idx - reid_res[index if reid_res[index] == idx else index - 1 if index - 1 >= 0 else index]) > 4:
                self.final_res.append(idx)
            else:
                self.final_res.append(reid_res[index if reid_res[index] == idx else index - 1 if index - 1 >= 0 else index])
        return self.final_res


def parse_args():
    parser = argparse.ArgumentParser("For Round Cut")
    parser.add_argument(
        "--video_path",
        default="round_cut_demo.mp4",
        type=str,
    )

    parser.add_argument(
        "--output_path",
        default="game_output/result.json",
        type=str,
    )

    parser.add_argument(
        "--round_clips",
        default="round_clips.json",
        type=str,
    )

    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )

    parser.add_argument(
        "--still-threshold", default=1, help="still person tracker's threshold"
    )
    
    parser.add_argument(
        "--still-frames", default=4, help="still person tracker's threshold"
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

    # OCR Region
    parser.add_argument(
        "--y_start",
        # default=615,
        default=620,
        help="Starting position of the Y-axis of the scoreboard",
    )

    parser.add_argument(
        "--y_end",
        # default=680,
        default=687,
        help="Ending position of the Y-axis of the scoreboard",
    )

    parser.add_argument(
        "--x_start",
        # default=316,
        default=319,
        help="Starting position of the X-axis of the scoreboard",
    )

    parser.add_argument(
        "--x_end",
        # default=356,
        default=355,
        help="Ending position of the X-axis of the scoreboard",
    )

    # yolo args
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
        "--query_path",
        default="fast_reid/query",
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
    score_cutter = ScoreCut(args)

    # load round_clips json file
    with open(args.round_clips, 'r') as file:
        round_clips = json.load(file)

    score_res=[]

    # 遍历每一局
    for clip in round_clips["clips"]:
        print("clip: {}".format(clip))
        score_cutter.load_video(args.video_path, clip[0], clip[1])
        with torch.no_grad():
            score_cutter.detect()
        print("ocr_res: {}".format(score_cutter.get_ocr_res()))
        print("reid_res: {}".format(score_cutter.get_still_tracker_res()))
        final_res = score_cutter.get_final_res(score_cutter.get_ocr_res(False),score_cutter.get_still_tracker_res(False))
        filter_values(final_res,5)
        print("filter_final_res: {}".format(final_res))
        score_res.append(final_res)
        print(score_cutter.tracker.results)

    print("score_res: {}".format(score_res))

    print("game_res: {}".format(score_cutter.tracker.results))

    last_res = score_cutter.tracker.results[-1]
    if last_res['end_frame'] == -1:
        last_res['end_frame'] = score_cutter.total_frames

    with open(args.output_path, 'w') as file:
        json.dump(score_cutter.tracker.results, file)

