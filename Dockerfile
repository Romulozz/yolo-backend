# yolo-backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias del sistema para Pillow/OpenCV/Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependencias
COPY server/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copia el backend (incluye /models con el .pt)
COPY server /app

# Render pasa $PORT; exponemos 8000 por convención
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
