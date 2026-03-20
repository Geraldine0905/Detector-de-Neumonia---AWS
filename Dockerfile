# ── Stage: producción ────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema para OpenCV + PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python (capa cacheada si requirements.txt no cambia)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir gunicorn psycopg2-binary

# Copiar código fuente
COPY . /app

# Crear carpetas de uploads/heatmaps en tiempo de build
RUN mkdir -p ui/static/uploads ui/static/heatmaps

# El modelo se monta como volumen externo en /app/models
VOLUME ["/app/models"]

EXPOSE 8000

# Gunicorn: 2 workers síncronos (TensorFlow no es thread-safe con múltiples workers)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "ui.main:app"]
