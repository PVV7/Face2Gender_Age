import numpy as np
import onnx
import onnxruntime
from ultralytics import YOLO
import dill
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


model = YOLO(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\runs\detect\train\weights\best.pt')
source = r'D:\\Plotnikov\\DS\\DS_projects\\Face2Gender_Age\\Detector\\Photo\\1.jpg'

res = model(source)
print(res)
# img = res[0].orig_img
#
# boxes = res[0].boxes.cpu().numpy()
# xyxys = boxes.xyxy
# xyxys = xyxys.tolist()[0]

# image = cv2.imread(source)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# image = cv2.rectangle(image, (int(xyxys[0]), int(xyxys[1])),
#                       (int(xyxys[2]), int(xyxys[3])),
#                       (255, 0, 0), 2
#                       )
#
# plt.imshow(image)
# plt.show()




# model = YOLO(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\runs\detect\train\weights\best.pt')
# model.export(format='onnx')

# path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\detector\detector.onnx'
#
# onnx_model = onnx.load(path)
#
# onnx.checker.check_model(onnx_model)
#




# # path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\onnx_models\best.onnx'
# path = r'detector/detector.onnx'
# img = r'D:\\Plotnikov\\DS\\DS_projects\\Face2Gender_Age\\Detector\\Photo\\7.jpg'
#
# onnx_model = YOLO(path)
# result = onnx_model.predict(img, task='detect')
#
# original_image = result[0].orig_img
#
# boxes = result[0].boxes.cpu().numpy()
# xyxys = boxes.xyxy
# xyxys = xyxys.tolist()[0]
#
# image = cv2.imread(img)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# image = cv2.rectangle(image,
#                       (int(xyxys[0]), int(xyxys[1])),
#                       (int(xyxys[2]), int(xyxys[3])),
#                       (255, 0, 0),
#                       2)
#
# plt.imshow(image)
# plt.show()