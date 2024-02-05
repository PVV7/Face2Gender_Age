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
from detector.detector import YoloModel

# model = YOLO(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\runs\detect\train\weights\best.pt')
# source = r'D:\\Plotnikov\\DS\\DS_projects\\Face2Gender_Age\\Detector\\Photo\\1.jpg'
#
# res = model(source)
# print(res)
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



# Convert to ONNX
# model = YOLO(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.pt')
# model.export(format='onnx')
#
# path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.onnx'
#
# onnx_model = onnx.load(path)
#
# onnx.checker.check_model(onnx_model)

# model = YOLO(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.pt')
# x = torch.rand(1, 3, 640, 640).to('cpu')
# device = 'cpu'
# torch.onnx.export(model,                                # model being run
#                   x,                                    # model input (or a tuple for multiple inputs)
#                   "custom_convert_Yolov8n.onnx",           # where to save the model (can be a file or file-like object)
#                   input_names = ['images'],              # the model's input names
#                   output_names = ['output'])




# path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\onnx_models\best.onnx'


# path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.onnx'
path = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.pt'
img = r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\Photo\14.jpg'

# model = YoloModel(path, img)
onnx_model = YOLO(path)


result = onnx_model.predict(img, task='detect')
kps = result[0].keypoints.cpu().numpy()
print(result)
kp_xys = kps.xy
kp_xys = kps.xy.tolist()
print(kp_xys)

# model.crop_obj(path=r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\croped_image')

boxes = result[0].boxes.cpu().numpy()
xyxys = boxes.xyxy.astype(int)

xyxys = xyxys.tolist()

image = cv2.imread(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



for xyxy in xyxys:
    image = cv2.rectangle(image,
                          (xyxy[0], xyxy[1]),
                          (xyxy[2], xyxy[3]),
                          (255, 0, 0),
                          30)

for points_group in kp_xys:
    for xy in points_group:
        image = cv2.circle(image, (int(xy[0]), int(xy[1])), 10, (0, 255, 0), thickness=-1)





plt.imshow(image)
plt.show()




