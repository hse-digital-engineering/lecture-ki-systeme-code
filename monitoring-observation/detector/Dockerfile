FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir uv && uv pip install --system -r pyproject.toml

# Pre-download YOLO weights so container startup is fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

COPY detector.py .

CMD ["python", "detector.py"]
