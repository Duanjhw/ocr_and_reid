import numpy as np
import torch
import cv2


import sys

sys.path.append("./fast_reid")
sys.path.append("./fast_reid/demo")
from fast_reid.demo.demo import reid_feature


__all__ = ["DeepReid"]


class DeepReid(object):
    def __init__(self, args):
        self.extractor = reid_feature(args=args)

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def get_features(self, bbox_xywh, ori_img):
        im_crops = []
        self.height, self.width = ori_img.shape[:2]
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]

            im = im[:, :, ::-1]  # reid 前处理
            im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_CUBIC)
            im_crops.append(
                torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None]
            )
        if im_crops:
            features = self.extractor.run_on_batch(im_crops)
        else:
            features = np.array([])
        return features

    def get_extractor(self):
        return self.extractor


def get_reid(args):
    return reid_feature(args=args)
