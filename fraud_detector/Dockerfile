# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Директория для логов
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Установка зависимостей (inference только на CPU)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 --retries 10 -r requirements.txt

# Код сервиса и артефакты модели
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "./app/app.py"]
