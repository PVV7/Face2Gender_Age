from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os


class OnnxModel(object):
    def __init__(self, path):
        self._path = path

        self._model = YOLO(self._path)
        self._predict = self._model.predict(self._image, task='detect')

    @property
    def get_model_detector(self):
        pass










