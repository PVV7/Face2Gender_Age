import os
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.exceptions import HTTPException
import uvicorn
from project.utils.output_utils import convert_data
from project.utils.predict_image import predict_image

from project.detector.detector import YoloModel
from project.aligner.aligner import Aligner
from project.classificator import Classificator
import cv2



UPLOAD_DIR = Path() / 'test'
# UPLOAD_DIR = Path() / '/usr/project/upload_img'

app = FastAPI()

@app.get('/')
async def mainpage() -> str:
    # return FileResponse('/usr/project/www/public/index.html')
    return FileResponse('project/server/index.html')



@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):

    # Get the file size (in bytes)
    file.file.seek(0, 2)
    file_size = file.file.tell()

    # move the cursor back to the beginning
    await file.seek(0)

    if file_size > 5 * 1024 * 1024:
        # more than 5 MB
        raise HTTPException(status_code=400, detail="Размер файла слишком большой. Загрузите файл менее 5 MB")

    # check the content type (MIME type)
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Данное разрешение не подходит")

    data = await file.read()
    save_to = UPLOAD_DIR / file.filename

    with open(save_to, 'wb') as f:
        f.write(data)

    result = predict_image(save_to)
    result = convert_data(result, to_str=True)

    return {"result": result}



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)








