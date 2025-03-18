FROM python:3.11.0-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

RUN pip install --default-timeout=100 -r requirements.txt