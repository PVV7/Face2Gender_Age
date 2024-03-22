FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim-2024-03-04

WORKDIR /usr/project

COPY project project
COPY main.py main.py

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /usr/project
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p upload_img
RUN mkdir -p www/public

COPY project/web/index.html /usr/project/www/public

CMD ["python", "main.py"]

