FROM python:3.12

ARG WORK_DIR
WORKDIR $WORK_DIR

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
COPY . .