import numpy as np
import onnxruntime
import cv2
import os
from typing import List, Tuple, Dict

#TODO: 1) рассмотреть другой алгоритм, который работает с np.array (_postprocess) \
#      2) подписать все методы в классе, что в них приходит и что они выдают

class YoloModel(object):

    def __init__(self,
                 image: np.array,
                 onnx_path: str,
                 input_size=(640, 640),
                 box_score=0.7,
                 iou_threshold=0.45
                 ):

        assert onnx_path.endswith('.onnx'), f'Only .onnx files are supported: {onnx_path}'
        assert os.path.exists(onnx_path), f'model not found: {onnx_path}'

        self.image = image
        self.ort_sess = onnxruntime.InferenceSession(onnx_path)
        print('input info: ', self.ort_sess.get_inputs()[0])
        print('output info: ', self.ort_sess.get_outputs()[0])
        self.input_size = input_size
        self.box_score = box_score
        self.iou_threshold = iou_threshold

    def preprocess(self) -> Tuple:
        img = self.image
        input_w, input_h = self.input_size
        if len(img.shape) == 3:
            padded_img = np.ones((input_w, input_h, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114
        r = min(input_w / img.shape[0], input_h / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        img_for_view = padded_img.copy()  # временная переменная для отладки
        # (H, W, C) BGR -> (C, H, W) RGB (формат для подачи изображения в ONNX)
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, img_for_view, r # убрать временную переменную (img_for_view)

    def _postprocess(self, output: List[np.ndarray], ratio) -> Dict: # рассмотреть другой алгоритм, который работает с np.array
        out = output[0][0]
        class_index = out[4]
        box_x = out[0]
        box_y = out[1]
        box_width = out[2]
        box_height = out[3]

        kps = out[5:]

        landmarks = [obj for obj in zip(*kps)]

        box_and_points = [obj for obj in zip(class_index,
                                             box_x,
                                             box_y,
                                             box_width,
                                             box_height,
                                             landmarks
                                             )
                          ]

        box_and_points = [obj for obj in box_and_points if obj[0] > self.box_score]
        box_and_points = self._NMS(box_and_points, iou_thresh=self.iou_threshold)

        result = [{'score': np.array(obj[0]),
                   'boxes': np.array(obj[1:5]) / ratio,
                   'kpts': np.array(obj[5]) / ratio}
                  for obj in box_and_points] # временный result, позже надо убрать np.array во всех значения по ключам
        return result

    def detect(self):
        img, img_for_view, ratio = self.preprocess() # убрать временную переменную (img_for_view)
        img = img[None, :] / 255
        ort_input = {self.ort_sess.get_inputs()[0].name: img}
        output = self.ort_sess.run(None, ort_input)
        result = self._postprocess(output, ratio)
        return result

    def _NMS(self, box_and_points, iou_thresh=0.45):
        remove_flags = [False] * len(box_and_points)
        keep_boxes = []

        for i, ibox in enumerate(box_and_points):
            if remove_flags[i]:
                continue

            box = ibox[1:5]

            keep_boxes.append(ibox)

            for j in range(i + 1, len(box_and_points)):
                jbox = box_and_points[j]
                jbox = jbox[1:5]

                if self._iou(box, jbox) > iou_thresh:
                    remove_flags[j] = True

        return keep_boxes

    def _xywh2xyxy(self, box_xywh):
        cx, cy, w, h = box_xywh

        left_x = cx - w * 0.5
        top_y = cy - h * 0.5
        right_x = cx + w * 0.5
        bottom_y = cy + h * 0.5

        return left_x, top_y, right_x, bottom_y

    def _iou(self, box1, box2):
        box1_coord = self._xywh2xyxy(box1)
        box2_coord = self._xywh2xyxy(box2)

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

    def draw(self):
        points = self.detect()
        for point in points:
            box = point['boxes']
            kpts = point['kpts']

            left_x, left_y, right_x, right_y = self._xywh2xyxy(box)

            image = cv2.rectangle(self.image,
                                  (int(left_x), int(left_y)),
                                  (int(right_x), int(right_y)),
                                  (255, 0, 0),
                                  1)

            for i in range(0, len(kpts), 3):
                image = cv2.circle(image, (int(kpts[i]), int(kpts[i + 1])), 5, (0, 255, 0), thickness=-1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image