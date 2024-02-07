from detector.detector import YoloModel
from aligner.aligner import Aligner
import cv2
import matplotlib.pyplot as plt



if __name__ == '__main__':

    image_path = r'project\photo\1.jpg'
    image = cv2.imread(image_path)

    ONNX_model = r'project\detector\weights\yolov8n-face.onnx'
    model = YoloModel(ONNX_model)

    img = model.draw(image)
    plt.imshow(img)
    plt.show()


