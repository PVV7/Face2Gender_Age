from project.detector.detector import YoloModel
from project.aligner.aligner import Aligner
from project.classificator import Classificator
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # read image
    image_path = r'project\photo\6.png'
    image = cv2.imread(image_path)

    # detecting faces in the image
    ONNX_model = r'project\detector\weights\yolov8n-face.onnx'
    model_detector = YoloModel(ONNX_model)
    res = model_detector.detect(image)

    #align the resulting faces
    aligner = Aligner()
    align_images = aligner.align_faces(image, res)



    #classify persons by gender and age
    classificator = Classificator(r'project/classificator/weights/Face2AgeGender.onnx')
    res = classificator.classificate(align_images)

    print(res)
