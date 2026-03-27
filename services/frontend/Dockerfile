FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Only PostgreSQL client libs needed — no OpenCV, no TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ephemeral storage for uploads/heatmaps (mount EFS PVC in production)
RUN mkdir -p /tmp/uploads /tmp/heatmaps

EXPOSE 80

# 2 sync workers — safe because no global ML state
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "2", "--timeout", "120", "app:app"]
