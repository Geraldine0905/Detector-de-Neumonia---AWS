#!/bin/bash
# =============================================================================
# deploy.sh — Script de despliegue para instancias EC2
# Universidad Autónoma de Occidente — Computación en la Nube
# =============================================================================
# Uso:
#   En instancia 1 (us-east-2a, tiene PostgreSQL):
#     INSTANCE=1 bash deploy.sh
#
#   En instancia 2 (us-east-2b, solo app):
#     INSTANCE=2 DB_HOST=<IP_PRIVADA_INSTANCIA_1> bash deploy.sh
# =============================================================================

set -e

# ── Configuración — editar antes de desplegar ─────────────────────────────────
DOCKERHUB_USER="lamarck558"          # Tu usuario de Docker Hub
IMAGE_NAME="pneumoscan"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

POSTGRES_DB="${POSTGRES_DB:-neumonia}"
POSTGRES_USER="${POSTGRES_USER:-neumonia_user}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-changeme}"   # cambiar en producción
DB_HOST="${DB_HOST:-10.0.133.163}"       # IP privada instancia 1 (solo para instancia 2)

INSTANCE="${INSTANCE:-1}"             # 1 o 2
COMPOSE_FILE="docker-compose.instance${INSTANCE}.yml"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================="
echo "  PneumoScan — Deploy en instancia ${INSTANCE}"
echo "============================================="

# 1. Instalar Docker si no está instalado
if ! command -v docker &> /dev/null; then
    echo "[1/5] Instalando Docker..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    echo "[1/5] Docker instalado."
else
    echo "[1/5] Docker ya instalado: $(docker --version)"
fi

# 2. Crear usuario sin privilegios root para ejecutar contenedores
if ! id "appuser" &> /dev/null; then
    echo "[2/5] Creando usuario 'appuser'..."
    sudo useradd -m -s /bin/bash appuser
    sudo usermod -aG docker appuser
    echo "[2/5] Usuario 'appuser' creado y agregado al grupo docker."
else
    echo "[2/5] Usuario 'appuser' ya existe."
fi

# 3. Crear directorio de la app y copiar archivos
echo "[3/5] Preparando directorio /home/appuser/pneumoscan..."
sudo mkdir -p /home/appuser/pneumoscan/models
sudo cp "${COMPOSE_FILE}" /home/appuser/pneumoscan/docker-compose.yml

# Crear archivo .env con las variables de entorno
sudo tee /home/appuser/pneumoscan/.env > /dev/null <<EOF
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
DB_HOST=${DB_HOST}
EOF

sudo chown -R appuser:appuser /home/appuser/pneumoscan
echo "[3/5] Directorio listo."

# 4. Descargar imagen desde Docker Hub
echo "[4/5] Descargando imagen ${FULL_IMAGE}..."
sudo -u appuser docker pull "${FULL_IMAGE}"
echo "[4/5] Imagen descargada."

# 5. Levantar contenedores con docker compose
echo "[5/5] Levantando contenedores..."
cd /home/appuser/pneumoscan
sudo -u appuser docker compose up -d

echo ""
echo "============================================="
echo "  Despliegue completado en instancia ${INSTANCE}"
echo "  App disponible en: http://$(hostname -I | awk '{print $1}'):8000"
echo "  Health check:      http://$(hostname -I | awk '{print $1}'):8000/health"
echo "============================================="
