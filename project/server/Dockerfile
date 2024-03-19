FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim-2024-03-04

WORKDIR /usr/project
COPY main.py main.py

COPY requirements.txt /usr/project
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/project/www/public
COPY index.html /usr/project/www/public

RUN mkdir -p /usr/project/upload_img

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


