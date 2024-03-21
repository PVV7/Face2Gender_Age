import numpy as np
import onnxruntime
import cv2
import os
from typing import List, Tuple, Dict, Any
from project.utils import xywh2xyxy, NMS


class Yolov6(object):

    def __init__(self,
                 onnx_path: str,
                 input_size=(640, 640),
                 box_score=0.3,
                 iou_threshold=0.45
                 ):

        assert onnx_path.endswith('.onnx'), f'Only .onnx files are supported: {onnx_path}'
        assert os.path.exists(onnx_path), f'model not found: {onnx_path}'


        self.ort_sess = onnxruntime.InferenceSession(onnx_path)
        print('Detector')
        print('input info: ', self.ort_sess.get_inputs()[0])
        print('output info: ', self.ort_sess.get_outputs()[0])
        self.input_size = input_size
        self.box_score = box_score
        self.iou_threshold = iou_threshold

    def _preprocess(self, image: np.ndarray) -> Tuple:

        img = image

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
        # (H, W, C) BGR -> (C, H, W) RGB (формат для подачи изображения в ONNX)
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        return padded_img, r

    def _postprocess(self,
                     output: List[np.ndarray],
                     ratio: float) -> List[Dict]:

        predict = output[0].squeeze(0)
        predict = predict[predict[:, 15] > self.box_score, :]

        scores = predict[:, 15]
        boxes = predict[:, 0:4] / ratio
        kpts = predict[:, 4:14] /ratio

        box_and_points = [obj for obj in zip(scores,
                                             boxes,
                                             kpts
                                             )
                          ]

        box_and_points = NMS(box_and_points, iou_thresh=self.iou_threshold)

        result = [{'score': obj[0],
                   'boxes': obj[1],
                   'kpts':  obj[2]}
                    for obj in box_and_points]

        return result

    def detect(self, image: np.ndarray) -> List[Dict]:

        assert image is not None, 'Image cannot be None'

        img, ratio = self._preprocess(image)
        img = img[None, :] / 255

        ort_input = {self.ort_sess.get_inputs()[0].name: img}
        output = self.ort_sess.run(None, ort_input)

        result = self._postprocess(output, ratio)
        assert len(result) > 0, 'There are no people in the image'
        return result


    def draw(self, image: np.ndarray) -> np.ndarray:

        points = self.detect(image)

        for point in points:
            box = point['boxes']
            kpts = point['kpts']
            left_x, left_y, right_x, right_y = xywh2xyxy(box)

            image = cv2.rectangle(image,
                                  (int(left_x), int(left_y)),
                                  (int(right_x), int(right_y)),
                                  (255, 0, 0),
                                  3)

            for i in range(0, len(kpts) - 1, 2):
                image = cv2.circle(image, (int(kpts[i]), int(kpts[i + 1])), 8, (0, 255, 0), thickness=-1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

