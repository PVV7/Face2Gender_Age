from detector.detector import YoloModel
import cv2
import matplotlib.pyplot as plt




if __name__ == '__main__':
    image = cv2.imread(r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\Photo\14.jpg')
    # plt.imshow(image)
    # plt.show()

    model = YoloModel('D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Face2Gender_Age\project\detector\weights\yolov8n-face.onnx')
    img, img_for_view, ratio = model.preprocess(image)




    new_img = model.draw(img_for_view)

    plt.imshow(new_img)
    plt.show()
