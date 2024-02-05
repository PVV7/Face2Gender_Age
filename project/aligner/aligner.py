import cv2
import numpy as np
from skimage import transform as trans

class Aligner(object):
    def __init__(self, model):
        self.model = model
        self.result = self.model.detect()
        self.image = self.model.image


    def _align(self):
        pass