#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/18
# @Author : zengwb

import torch
import numpy as np

from round_utils.general import (
     non_max_suppression, scale_coords)


def bbox_r(x1, y1, x2, y2):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min(x1, x2)
    bbox_top = min(y1, y2)
    bbox_w = abs(x1 - x2)
    bbox_h = abs(y1 - y2)
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def adjust_coordinates(ori_img, img, x1, y1, x2, y2):
    x1 = int(x1 * ori_img.shape[1] / img.shape[1])
    y1 = int(y1 * ori_img.shape[0] / img.shape[0])
    x2 = int(x2 * ori_img.shape[1] / img.shape[1])
    y2 = int(y2 * ori_img.shape[0] / img.shape[0])
    return x1, y1, x2, y2

class PersonDectect():
    def __init__(self, opt):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = opt.conf_thres  # confidence threshold (0-1)
        self.model.iou = opt.iou_thres  # NMS IoU threshold (0-1)
        self.model.classes = 0 
        self.model.agnostic_nms = opt.agnostic_nms

    def detect(self, img, im0s):
        # Inference
        new_img = img.transpose(1,2,0)
        pred = self.model(new_img)
        detections = pred.pandas().xyxy[0]
       
        # Process detections
        bbox_xywh = []
        xy = []
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
            x1, y1, x2, y2 = adjust_coordinates(im0s, new_img, x1, y1, x2, y2)
       
            x_c, y_c, bbox_w, bbox_h = bbox_r(x1, y1, x2, y2)
            obj = [x_c, y_c, bbox_w, bbox_h]
            bbox_xywh.append(obj)
            xy.append([x1, y1, x2, y2])
        return np.array(bbox_xywh),xy
