from project.detector.detector import YoloModel
from project.aligner.aligner import Aligner
from project.classificator import Classificator
import cv2
import matplotlib.pyplot as plt

import os
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



if __name__ == '__main__':
    #
    # # read image
    # image_path = r'project/photo/6.png'
    # image = cv2.imread(image_path)
    #
    # # detecting faces in the image
    # ONNX_model = r'project/detector/weights/yolov8n-face.onnx'
    # model_detector = YoloModel(ONNX_model)
    # res = model_detector.detect(image)
    #
    # #align the resulting faces
    # aligner = Aligner()
    # align_images = aligner.align_faces(image, res)
    #
    # #classify persons by gender and age
    # classificator = Classificator(r'project/classificator/weights/Face2AgeGender.onnx')
    # res = classificator.classificate(align_images)
    #
    # print(res)


    q = queue.Queue()


    class MyHandler(FileSystemEventHandler):
        @staticmethod
        def work(event):
            flag = False
            extension = ('.jpg', '.jpeg', '.png', 'JPEG')

            if not event.is_directory and event.src_path.endswith(extension):
                flag = True
            return flag

        def on_created(self, event):
            if self.work(event):
                temp = event.src_path.replace('\\', '/')
                # print(f"Файл {event.src_path} добавлен в очередь")
                # q.put(event.src_path)
                print(f"Файл {temp} добавлен в очередь")
                q.put(temp)


    path = 'test'
    observer = Observer()
    handler = MyHandler()

    observer.schedule(handler, path, recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            curr_image = q.get()
            observer.join(1)

            print(f'Обработка изображения {curr_image}')

            image_path = fr'{curr_image}'
            image = cv2.imread(image_path)

            # detecting faces in the image
            ONNX_model = r'project/detector/weights/yolov8n-face.onnx'
            model_detector = YoloModel(ONNX_model)
            res = model_detector.detect(image)

            # align the resulting faces
            aligner = Aligner()
            align_images = aligner.align_faces(image, res)

            # classify persons by gender and age
            classificator = Classificator(r'project/classificator/weights/Face2AgeGender.onnx')
            res = classificator.classificate(align_images)

            print(res)

            print(f'Удаление изображения {curr_image}')
            os.remove(curr_image)

    finally:
        observer.stop()
        observer.join()
