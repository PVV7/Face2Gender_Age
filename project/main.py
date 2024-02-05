from detector.detector import YoloModel
from aligner.aligner import Aligner
import cv2
import matplotlib.pyplot as plt

#TODO: сделать так, чтобы программа считывала jpg и png форматы


if __name__ == '__main__':
    image = cv2.imread(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\Photo\14.jpg')
    # plt.imshow(image)
    # plt.show()
    ONNX_model = 'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.onnx'

    model = YoloModel(image, ONNX_model)

    new_img = model.draw()
    plt.imshow(new_img)
    plt.show()
