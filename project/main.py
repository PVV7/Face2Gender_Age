from detector.detector import YoloModel
from aligner.aligner import Aligner
import cv2
import matplotlib.pyplot as plt

#TODO: разобраться с весами (неправильно лежат в папке на github)


if __name__ == '__main__':

    image_path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\Photo\14.jpg'

    image = cv2.imread(image_path)
    # plt.imshow(image)
    # plt.show()
    ONNX_model = 'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.onnx'

    model = YoloModel(image, ONNX_model)

    aligner = Aligner(model)

    res = aligner.align_faces()
    res = res[0]

    plt.imshow(res)
    plt.show()

    # new_img = model.draw()
    # plt.imshow(new_img)
    # plt.show()
