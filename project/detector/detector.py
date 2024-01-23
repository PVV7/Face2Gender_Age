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
        boxes = self._predict[0].boxes.cpu().numpy()
        xyxys = boxes.xyxy

        return xyxys.tolist()[0]

    def view(self):
        image = self._image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        xyxys = self.get_boxes()

        image = cv2.rectangle(image,
                              (int(xyxys[0]), int(xyxys[1])),
                              (int(xyxys[2]), int(xyxys[3])),
                              (255, 0, 0),
                              2)

        plt.imshow(image)
        plt.show()

