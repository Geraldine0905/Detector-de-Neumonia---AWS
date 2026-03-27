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

# Copy application, ML source modules and trained model
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Single worker with 4 threads — TensorFlow graph is thread-safe but not fork-safe.
# Use --preload so the model loads once before Gunicorn forks (if workers > 1).
# sync worker, no --preload: TF loads inside the worker after fork (fork-safe)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--worker-class", "sync", \
     "--workers", "1", \
     "--timeout", "300", \
     "app:app"]
