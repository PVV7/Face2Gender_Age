import os
from project.detector.YOLOv6 import Yolov6
from project.aligner.aligner import Aligner
from project.classificator import Classificator
import cv2
from project.utils.output_utils import convert_data

def predict_image(img_path):
    img_path = str(img_path)

    image_path = img_path
    image = cv2.imread(image_path)

    # detecting faces in the image
    ONNX_model = r'project/detector/weights/yolov6m_face.onnx'
    model_detector = Yolov6(ONNX_model)
    res = model_detector.detect(image)

    # align the resulting faces
    aligner = Aligner()
    align_images = aligner.align_faces(image, res)

    # classify persons by gender and age
    classificator = Classificator(r'project/classificator/weights/Face2AgeGender.onnx')
    res = classificator.classificate(align_images)


    os.remove(image_path)
    return res