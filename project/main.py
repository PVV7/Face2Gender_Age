from detector.detector import YoloModel
from aligner.aligner import Aligner
from classificator import Classificator
import cv2
import matplotlib.pyplot as plt

#:TODO обработать вариант с неправильным форматом изображения
#:TODO обработать вариант, когда на фото нет лиц
#:TODO добавить расшифровку информации после классификатора
#:TODO попробовать обучить более тяжелую модель (детектор)


if __name__ == '__main__':

    image_path = r'project\photo\1.jpg'
    image = cv2.imread(image_path)


    ONNX_model = r'project\detector\weights\yolov8n-face.onnx'
    model_detector = YoloModel(ONNX_model)
    res = model_detector.detect(image)

    img = model_detector.draw(image)
    # plt.imshow(img)
    # plt.show()
    aligner = Aligner()
    align_images = aligner.align_faces(image, res)


    classificator = Classificator(r'project/classificator/weights/Face2AgeGender.onnx')
    res = classificator.classificate(align_images)
    print(res)
