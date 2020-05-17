FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
