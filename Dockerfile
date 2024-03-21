#FROM python:3.10-slim
#
#WORKDIR /usr/project
#
#COPY project project
#COPY main.py main.py
#
#
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#
#COPY requirements.txt /usr/project
#RUN pip install --no-cache-dir -r requirements.txt
#RUN mkdir -p /usr/project/upload_img
#
#CMD ["python", "main.py"]
##CMD ["python", "test.py"]
#
##CMD cd /usr/local && python3 api.py

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim-2024-03-04

WORKDIR /usr/project

COPY project project
COPY test.py test.py

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /usr/project
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /usr/project/upload_img

RUN mkdir -p /usr/project/www/public
COPY ./project/server/index.html /usr/project/www/public


CMD ["python", "test.py"]

#CMD cd /usr/local && python3 api.py
