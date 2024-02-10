import onnxruntime
import cv2
from typing import List, Tuple, Dict, Any
import os
import numpy as np


class Classificator(object):
    def __init__(self,
                 onnx_path: str,
                 input_size=(112, 112),
                 batch_size=32
                 ):

        assert onnx_path.endswith('.onnx'), f'Only .onnx files are supported: {onnx_path}'
        assert os.path.exists(onnx_path), f'model not found: {onnx_path}'

        self.ort_sess = onnxruntime.InferenceSession(onnx_path)
        print('Classificator')
        print('input info: ', self.ort_sess.get_inputs()[0])
        print('output info: ', self.ort_sess.get_outputs()[0])

        self.onnx_path = onnx_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.shape = self.ort_sess.get_inputs()[0].shape


    def _split_list(self,
                    list_crops: List[np.ndarray],
                    batch: int) -> List[List[np.ndarray]]:
        split_list = []
        for i in range(0, len(list_crops), batch):
            split_list.append(list_crops[i:i+batch])

        return split_list

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_AREA)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32)

        return img

    def classificate(self, crops: List[np.ndarray]) -> List[Any]:
        list_crops = list(map(self._preprocess, crops))
        split_list = self._split_list(list_crops, self.batch_size)

        shape_from_ONNX = self.ort_sess.get_inputs()[0].shape
        matrix_images = np.ones((shape_from_ONNX))
        matrix_images = matrix_images.astype('float32')
        print(matrix_images.shape)

        last_index = len(split_list[-1])
        results = []

        for i, images in enumerate(split_list):
            for j, img in enumerate(images):
                matrix_images[j] = img
            output = self.ort_sess.run(None, {'input': matrix_images})
            results.append(output)

        res = self._postprocess(results, last_index)
        print('result', type(res))
        return res

    def _postprocess(self,
                     results: List[np.ndarray],
                     last_index: int) -> List[np.ndarray]:

        if len(results) <= 1:
            temp = []
            for obj in results:
                for i in range(len(obj)):
                    temp.append(obj[i][:last_index])
            return temp
        else:
            last_images = []
            for obj in results[-1]:
                last_images.append(obj[:last_index])

        result = results[:-1] + [last_images]
        return result




