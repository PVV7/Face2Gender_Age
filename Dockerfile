FROM python:3.10-slim

WORKDIR /usr/project

COPY project project
COPY main.py main.py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /usr/project
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
CMD cd /usr/local && python3 api.py

