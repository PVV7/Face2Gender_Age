from detector.detector import OnnxModel


if __name__ == '__main__':
    path_onnx = r'detector/detector.onnx'
    image = r'D:\\Plotnikov\\DS\\DS_projects\\Face2Gender_Age\\Detector\\Photo\\4.jpg'
    model = OnnxModel(path_onnx, image)

    boxes = model.view_model()[0]
    print(boxes)
    # boxes = boxes.boxes
    # print(boxes.xyxy.tolist())
    # model.crop_obj(path=r'D:\Plotnikov\DS\DS_projects\Face2Gender_Age\Detector\croped_image')



