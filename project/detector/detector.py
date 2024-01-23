from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


class OnnxModel(object):
    def __init__(self, path, image):
        self._path = path
        self._image = image
        self._model = YOLO(self._path)
        self._predict = self._model.predict(self._image, task='detect')

    def predict(self):
        # TODO: реализовать вывод координат боксов и лендмарок(для афинных преобразований)
        pass

    def get_boxes(self):
        boxes = self._predict[0].boxes
        if boxes is None:
            return []

        boxes = boxes.cpu().numpy()
        xyxys = boxes.xyxy

        return xyxys.tolist()

    def crop_obj(self):
        boxes = self.get_boxes()
        if len(boxes) == 0:
            return f'лиц нет'

        image = cv2.imread(self._image)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop_obj = image[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite('croped_image_' + str(i) + '.jpg', crop_obj)


    def view_model(self): # временный метод, показывает результаты работы модели
        return self._predict


    def view(self): #временный метод для вывода результата на изображения
        image = self._image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        xyxys = self.get_boxes()

        for xyxy in xyxys:
            image = cv2.rectangle(image,
                                  (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])),
                                  (255, 0, 0),
                                  2)

        plt.imshow(image)
        plt.show()

