import numpy as np
from typing import List, Tuple, Dict, Any


def xywh2xyxy(box_xywh: np.array) -> Tuple:
    cx, cy, w, h = box_xywh

    left_x = cx - w * 0.5
    top_y = cy - h * 0.5
    right_x = cx + w * 0.5
    bottom_y = cy + h * 0.5

    return left_x, top_y, right_x, bottom_y


def NMS(box_and_points: List[Any],
        iou_thresh=0.45) -> List[Any]:

    remove_flags = [False] * len(box_and_points)
    keep_boxes = []

    for i, ibox in enumerate(box_and_points):
        if remove_flags[i]:
            continue

        box = ibox[1]

        keep_boxes.append(ibox)

        for j in range(i + 1, len(box_and_points)):
            jbox = box_and_points[j]
            jbox = jbox[1]

            if iou(box, jbox) > iou_thresh:
                remove_flags[j] = True

    return keep_boxes


def iou(box1: np.array,
        box2: np.array) -> Any:
    box1_coord = xywh2xyxy(box1)
    box2_coord = xywh2xyxy(box2)

    box1_area = (box1_coord[2] - box1_coord[0]) * (box1_coord[3] - box1_coord[1])
    box2_area = (box2_coord[2] - box2_coord[0]) * (box2_coord[3] - box2_coord[1])

    intersection = np.array([
        max(box1_coord[0], box2_coord[0]),
        max(box1_coord[1], box2_coord[1]),
        min(box1_coord[2], box2_coord[2]),
        min(box1_coord[3], box2_coord[3]),
    ])

    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou
