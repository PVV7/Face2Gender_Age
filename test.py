import os
from fastapi import FastAPI, UploadFile, APIRouter
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
import uvicorn
from datetime import datetime
from fastapi import BackgroundTasks
from pydantic import BaseModel

from project.detector.detector import YoloModel
from project.aligner.aligner import Aligner
from project.classificator import Classificator
import cv2
import matplotlib.pyplot as plt

import logging

UPLOAD_DIR = Path() / 'test'
# UPLOAD_DIR = Path() / '/usr/project/upload_img'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Person(BaseModel):
    age: int
    gender: str



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_image(img_path):
    img_path = str(img_path)
    logger.info(f'Картинка {img_path} обрабатывается')

    image_path = img_path
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

    # print(res)
    logger.info(f'Картинка {img_path} Результат {res} ')
    logger.info(f'Картинка {img_path} удалена')
    os.remove(image_path)
    return res

@app.get('/')
async def mainpage() -> str:
    # return FileResponse('/usr/project/www/public/index.html')
    return FileResponse('project/server/index.html')


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile, background_tasks: BackgroundTasks):

    # Get the file size (in bytes)
    file.file.seek(0, 2)
    file_size = file.file.tell()

    # move the cursor back to the beginning
    await file.seek(0)

    if file_size > 5 * 1024 * 1024:
        # more than 10 MB
        raise HTTPException(status_code=400, detail="File too large")

    # check the content type (MIME type)
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await file.read()
    save_to = UPLOAD_DIR / file.filename

    with open(save_to, 'wb') as f:
        f.write(data)
        now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # do something with the valid file
    logger.info(f'Картинка {save_to} добавлена в очередь')
    background_tasks.add_task(predict_image, save_to)


    return {"filename": file.filename, "date": current_time}



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)








