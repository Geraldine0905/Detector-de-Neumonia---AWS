FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# OpenCV headless dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application and ML source modules
COPY app.py .
COPY src/ ./src/

# Model is never baked into the image — always mounted as a volume/PVC
VOLUME ["/app/models"]

EXPOSE 8000

# Single worker with 4 threads — TensorFlow graph is thread-safe but not fork-safe.
# Use --preload so the model loads once before Gunicorn forks (if workers > 1).
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--preload", \
     "app:app"]
