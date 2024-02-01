import numpy as np
import onnxruntime
import cv2
import os


class YoloModel(object):

    def __init__(self,
                 onnx_path: str,
                 input_size=(640, 640),
                 box_score=0.7,
                 iou_threshold=0.45
                 ):

        assert onnx_path.endswith('.onnx'), f'Only .onnx files are supported: {onnx_path}'
        assert os.path.exists(onnx_path), f'model not found: {onnx_path}'

        self.ort_sess = onnxruntime.InferenceSession(onnx_path)
        print('input info: ', self.ort_sess.get_inputs()[0])
        print('output info: ', self.ort_sess.get_outputs()[0])
        self.input_size = input_size
        self.box_score = box_score
        self.iou_threshold = iou_threshold

    def _preprocess(self, img: np.array):
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
        # (H, W, C) BGR -> (C, H, W) RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _postprocess(self):
        pass

    def detect(self, img):
        img, ratio = self._preprocess(img)
        ort_input = {self.ort_sess.get_inputs()[0].name: img}
        output = self.ort_sess.run(None, ort_input)
        # result = self._postprocess()
        return

