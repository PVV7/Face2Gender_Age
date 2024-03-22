from project.aligner.aligner import Aligner
from project.classificator import Classificator
import cv2
import matplotlib.pyplot as plt
import onnxruntime
from project.detector.YOLOv6 import Yolov6
from project.utils.output_utils import convert_data
import os


if __name__ == "__main__":

    image_path = 'project/photo/6.png'

    image = cv2.imread(image_path)

    # detecting faces in the image
    ONNX_model = r'project/detector/ONNX/yolov6m_face.onnx'
    model_detector = Yolov6(ONNX_model)
    res = model_detector.detect(image)

    # align the resulting faces
    aligner = Aligner()
    align_images = aligner.align_faces(image, res)

    # classify persons by gender and age
    classificator = Classificator(r'project/classificator/ONNX/Face2AgeGender.onnx')
    res = classificator.classificate(align_images)

    res = convert_data(res)
    print(res)


