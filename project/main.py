from detector.detector import OnnxModel


if __name__ == '__main__':

    path_onnx = r'detector/detector.onnx'
    image = r'D:\\Plotnikov\\DS\\DS_projects\\Face2Gender_Age\\Detector\\Photo\\7.jpg'
    model = OnnxModel(path_onnx, image)

    model.view()


