<div align="center">

# 🫁 PneumoScan

### Sistema de Apoyo al Diagnóstico de Neumonía mediante Deep Learning

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/lamarck558/pneumoscan)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ALB-FF9900?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)

🌐 **Demo en vivo:** http://pneumonia-alb-2053601714.us-east-2.elb.amazonaws.com

</div>

---

## ¿Qué es PneumoScan?

PneumoScan es una aplicación web de apoyo al diagnóstico médico que analiza radiografías de tórax usando una **Red Neuronal Convolucional (CNN)** entrenada con TensorFlow. En segundos, el sistema clasifica la imagen en una de tres categorías y genera un mapa de calor que le muestra al médico exactamente qué parte de la radiografía influyó en el diagnóstico.

El proyecto está desplegado en **AWS con alta disponibilidad**: dos instancias EC2 en distintas zonas de disponibilidad detrás de un Application Load Balancer, con una base de datos PostgreSQL que registra cada predicción realizada.

### Funcionalidades principales

- **Clasificación en 3 categorías:** Neumonía Bacteriana, Neumonía Viral, Pulmón Normal
- **Explicabilidad con Grad-CAM:** mapa de calor superpuesto que indica qué regiones determinaron el diagnóstico
- **Soporte multi-formato:** acepta imágenes DICOM (formato médico estándar), JPEG y PNG
- **Persistencia en base de datos:** cada predicción queda registrada en PostgreSQL con timestamp
- **Exportación de reportes:** descarga del diagnóstico en PDF con imagen y heatmap, o en CSV
- **Alta disponibilidad en AWS:** dos réplicas de la app en zonas de disponibilidad distintas

---

## Índice

1. [Cómo funciona el modelo](#cómo-funciona-el-modelo)
2. [Arquitectura del código](#arquitectura-del-código)
3. [Arquitectura AWS](#arquitectura-aws)
4. [Ejecución local (sin Docker)](#ejecución-local-sin-docker)
5. [Ejecución local con Docker](#ejecución-local-con-docker)
6. [Despliegue en AWS](#despliegue-en-aws)
7. [Base de datos](#base-de-datos)
8. [API y endpoints](#api-y-endpoints)
9. [Pruebas de alta disponibilidad](#pruebas-de-alta-disponibilidad)
10. [Variables de entorno](#variables-de-entorno)
11. [Stack tecnológico](#stack-tecnológico)

---

## Cómo funciona el modelo

Cuando el usuario sube una radiografía, el sistema ejecuta el siguiente pipeline de inferencia:

```
Imagen (DICOM / JPEG / PNG)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   read_img.py                        │
│  • DICOM → extrae pixel_array → normaliza a uint8   │
│  • JPEG/PNG → lee con OpenCV → convierte a RGB       │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│                preprocess_img.py                     │
│  1. Resize a 512×512 px                             │
│  2. Conversión a escala de grises                   │
│  3. CLAHE (mejora contraste local, clipLimit=2.0)   │
│  4. Normalización a [0, 1]                          │
│  5. Reshape → tensor (1, 512, 512, 1)               │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              CNN — conv_MLP_84.h5                    │
│  Salida: vector de probabilidades [bacteriana,       │
│          normal, viral]                              │
│  Clase predicha: argmax del vector                  │
└─────────────┬───────────────┬───────────────────────┘
              │               │
              ▼               ▼
    Etiqueta + confianza    grad_cam.py
                            • GradientTape sobre capa conv10_thisone
                            • Pesos = promedio de gradientes por canal
                            • CAM = combinación lineal de activaciones
                            • Heatmap JET superpuesto sobre la imagen original
                                    │
                                    ▼
                          Resultado en la interfaz
                          + guardado en PostgreSQL
```

### Grad-CAM — ¿por qué es importante?

El modelo no es una caja negra. La técnica **Gradient-weighted Class Activation Mapping (Grad-CAM)** calcula, para cada predicción, qué regiones espaciales de la radiografía activaron más la neurona que tomó la decisión. El resultado es un mapa de calor en escala de colores JET (azul frío → rojo caliente) superpuesto a la radiografía original, lo que permite al médico validar si el modelo está mirando las zonas correctas.

---

## Arquitectura del código

```
uaoneumonia/
│
├── ui/                             ← Capa de presentación
│   ├── main.py                     ← Flask app: rutas, DB, lógica de negocio
│   ├── templates/
│   │   └── index.html              ← Interfaz web (tema oscuro, drag & drop)
│   └── static/
│       ├── uploads/                ← Imágenes subidas por el usuario
│       └── heatmaps/               ← Mapas de calor generados
│
├── src/                            ← Capa de dominio / lógica de IA
│   ├── app/
│   │   ├── integrator.py           ← Facade: orquesta todo el pipeline de inferencia
│   │   └── models/
│   │       └── load_model.py       ← Carga del modelo Keras (.h5)
│   ├── data/
│   │   └── read_img.py             ← Lectura de DICOM y JPEG/PNG → numpy array
│   ├── features/
│   │   └── preprocess_img.py       ← Resize, CLAHE, normalización → tensor
│   └── visualizations/
│       └── grad_cam.py             ← Generación del heatmap Grad-CAM
│
├── models/                         ← (NO incluido en el repo)
│   └── conv_MLP_84.h5              ← Modelo entrenado (~113 MB), montado como volumen
│
├── data/raw/                       ← Imágenes de ejemplo para pruebas
│   ├── DICOM/
│   └── JPG/
│
├── tests/                          ← Tests unitarios
│   ├── test_read_img.py
│   ├── test_preprocess_img.py
│   └── test_grad_cam.py
│
├── Dockerfile                      ← Imagen de producción (gunicorn, puerto 8000)
├── docker-compose.instance1.yml    ← Instancia 1: app + PostgreSQL
├── docker-compose.instance2.yml    ← Instancia 2: solo app (DB remota)
├── deploy.sh                       ← Script de despliegue automático en EC2
├── requirements.txt
└── README.md
```

### Patrón de diseño: Facade

El módulo `integrator.py` implementa el patrón **Facade**: expone un único método `predict(file_path)` que internamente coordina la lectura, el preprocesamiento, la inferencia y la generación del Grad-CAM. La UI solo necesita instanciar `PneumoniaDetector` y llamar a `predict()`.

```python
detector = PneumoniaDetector(model_path="models/conv_MLP_84.h5")
result = detector.predict("radiografia.dcm")

print(result.label)       # "bacteriana" | "normal" | "viral"
print(result.probability) # float 0–100
print(result.heatmap)     # numpy array (512, 512, 3) — imagen RGB
```

---

## Arquitectura AWS

```
                           INTERNET
                               │
                               ▼
              ┌────────────────────────────────┐
              │   Application Load Balancer     │
              │   pneumonia-alb (puerto 80)     │
              │   Health check: GET /health     │
              └──────────┬─────────────┬────────┘
                         │             │
            ┌────────────▼──┐    ┌─────▼─────────────┐
            │  EC2 — AZ a   │    │  EC2 — AZ b        │
            │  us-east-2a   │    │  us-east-2b        │
            │  t3.small     │    │  t3.small          │
            │               │    │                    │
            │ ┌───────────┐ │    │  ┌───────────┐    │
            │ │ neumonia_ │ │    │  │ neumonia_ │    │
            │ │ app:8000  │ │    │  │ app:8000  │    │
            │ └─────┬─────┘ │    │  └─────┬─────┘    │
            │       │       │    │        │           │
            │ ┌─────▼─────┐ │◄───┼────────┘           │
            │ │neumonia_db│ │  IP privada instancia 1  │
            │ │ postgres  │ │    │                    │
            │ │  :5432    │ │    │                    │
            │ └───────────┘ │    │                    │
            └───────────────┘    └────────────────────┘
              Subred privada        Subred privada
              10.0.128.0/20         10.0.144.0/20

VPC: 10.0.0.0/16
  Subred pública AZ-a: 10.0.0.0/20    ← NAT Gateway + nodo ALB
  Subred pública AZ-b: 10.0.16.0/20   ← nodo ALB
  Subred privada AZ-a: 10.0.128.0/20  ← Instancia 1 (app + DB)
  Subred privada AZ-b: 10.0.144.0/20  ← Instancia 2 (solo app)
```

### Decisiones de diseño

| Decisión | Razón |
|---|---|
| Instancias en **subredes privadas** | No expuestas directamente a internet — solo accesibles via ALB |
| **Un solo contenedor PostgreSQL** en instancia 1 | Datos centralizados; ambas instancias escriben en la misma DB |
| Modelo montado como **volumen externo** | El archivo .h5 de 113 MB no se incluye en la imagen Docker — se descarga por separado |
| **2 workers Gunicorn** (no más) | TensorFlow no es thread-safe con múltiples workers que comparten el grafo del modelo |
| `restart: always` en Docker Compose | Los contenedores se reinician solos si fallan o si la instancia EC2 reinicia |

### Security Groups (mínimos privilegios)

| Security Group | Aplicado a | Entrada permitida | Puerto |
|---|---|---|---|
| `pneumonia-lb-sg` | ALB | `0.0.0.0/0` (internet) | 80 |
| `pneumonia-app-sg` | EC2 instancias | `pneumonia-lb-sg` | 8000 |
| `pneumonia-app-sg` | EC2 instancias | `pneumonia-app-sg` | 5432 (DB) |
| `bastion-sg` | Bastion Host | Tu IP específica | 22 |
| `pneumonia-app-sg` | EC2 (SSH) | `bastion-sg` | 22 |

---

## Ejecución local (sin Docker)

### Requisitos

- Python 3.12
- El archivo del modelo: `models/conv_MLP_84.h5`

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/lamarck558/uaoneumonia.git
cd uaoneumonia

# 2. Crear entorno virtual
python -m venv venv

# En Windows:
.\venv\Scripts\Activate.ps1
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
pip install psycopg2-binary gunicorn

# 4. Colocar el modelo en la carpeta models/
mkdir -p models
# Copiar conv_MLP_84.h5 a models/

# 5. (Opcional) Levantar PostgreSQL local con Docker
docker run -d --name postgres-local \
  -e POSTGRES_DB=neumonia \
  -e POSTGRES_USER=neumonia_user \
  -e POSTGRES_PASSWORD=changeme \
  -p 5432:5432 postgres:15

# 6. Ejecutar la app
python ui/main.py

# La app estará disponible en: http://localhost:8000
```

> **Sin base de datos:** Si no configuras PostgreSQL, la app arranca igual. Los diagnósticos funcionan con normalidad; simplemente no se guardarán en la DB y verás un aviso en los logs.

---

## Ejecución local con Docker

Esta es la forma recomendada. Levanta la app y PostgreSQL en un solo comando.

### Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- El archivo del modelo: `models/conv_MLP_84.h5`

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/lamarck558/uaoneumonia.git
cd uaoneumonia

# 2. Colocar el modelo
mkdir -p models
# Copiar conv_MLP_84.h5 a models/

# 3. Levantar app + PostgreSQL
docker compose -f docker-compose.instance1.yml up -d

# 4. Verificar que todo está corriendo
docker compose -f docker-compose.instance1.yml ps
```

Salida esperada:

```
NAME           IMAGE                          STATUS
neumonia_app   lamarck558/pneumoscan:latest   Up (healthy)   0.0.0.0:8000->8000/tcp
neumonia_db    postgres:15                    Up (healthy)   0.0.0.0:5432->5432/tcp
```

```bash
# 5. Verificar el health check
curl http://localhost:8000/health
# → {"status": "ok"}

# 6. Abrir en el navegador
# http://localhost:8000
```

```bash
# Para detener los contenedores
docker compose -f docker-compose.instance1.yml down

# Para detener Y eliminar los datos de la DB
docker compose -f docker-compose.instance1.yml down -v
```

### Build propio de la imagen

Si modificas el código y quieres construir tu propia imagen:

```bash
docker build -t tu_usuario/pneumoscan:latest .
docker push tu_usuario/pneumoscan:latest
```

---

## Despliegue en AWS

### Requisitos previos en AWS

1. **VPC** con 4 subredes: 2 públicas (para el ALB y NAT Gateway) y 2 privadas (para las instancias EC2), una en cada zona de disponibilidad
2. **NAT Gateway** en la subred pública — permite a las instancias privadas descargar imágenes de Docker Hub
3. **2 instancias EC2** Amazon Linux 2023 (`t3.small` mínimo) en las subredes privadas
4. **Bastion Host** (opcional pero recomendado) para acceso SSH a las instancias privadas
5. **Bucket S3** con el modelo subido: `aws s3 cp models/conv_MLP_84.h5 s3://tu-bucket/conv_MLP_84.h5`
6. **Rol IAM** con `AmazonS3ReadOnlyAccess` asignado a las instancias EC2
7. **Security Groups** configurados según la tabla de la sección anterior
8. **Application Load Balancer** apuntando a las dos instancias

### Paso 1 — Subir el modelo a S3

```bash
# Desde tu máquina local (requiere AWS CLI configurado)
aws s3 cp models/conv_MLP_84.h5 s3://tu-bucket/conv_MLP_84.h5
```

### Paso 2 — Desplegar en Instancia 1 (us-east-2a)

Conéctate a la instancia 1 (vía bastion o SSM) y ejecuta:

```bash
# Descargar el script de despliegue
curl -O https://raw.githubusercontent.com/lamarck558/uaoneumonia/main/deploy.sh
curl -O https://raw.githubusercontent.com/lamarck558/uaoneumonia/main/docker-compose.instance1.yml

# Desplegar (ajusta la contraseña)
INSTANCE=1 POSTGRES_PASSWORD=tu_password_seguro bash deploy.sh
```

El script automáticamente:
- Instala Docker y Docker Compose Plugin
- Crea el usuario `appuser` sin privilegios root
- Agrega `appuser` al grupo `docker`
- Descarga la imagen `lamarck558/pneumoscan:latest` desde Docker Hub
- Levanta los contenedores `neumonia_app` y `neumonia_db`

Luego copia el modelo:

```bash
# Desde tu máquina local
scp -i tu-clave.pem -o ProxyJump=ubuntu@<IP_BASTION> \
    models/conv_MLP_84.h5 \
    ec2-user@<IP_PRIVADA_EC2_1>:/home/appuser/pneumoscan/models/
```

### Paso 3 — Desplegar en Instancia 2 (us-east-2b)

```bash
# Conectar a instancia 2 y ejecutar
curl -O https://raw.githubusercontent.com/lamarck558/uaoneumonia/main/deploy.sh
curl -O https://raw.githubusercontent.com/lamarck558/uaoneumonia/main/docker-compose.instance2.yml

# Reemplaza 10.0.133.163 con la IP PRIVADA real de la instancia 1
INSTANCE=2 DB_HOST=10.0.133.163 POSTGRES_PASSWORD=tu_password_seguro bash deploy.sh
```

Copiar el modelo a la instancia 2:

```bash
scp -i tu-clave.pem -o ProxyJump=ubuntu@<IP_BASTION> \
    models/conv_MLP_84.h5 \
    ec2-user@<IP_PRIVADA_EC2_2>:/home/appuser/pneumoscan/models/
```

### Paso 4 — Configurar el ALB

En **AWS Console → EC2 → Load Balancers**:

**Target Group:**
- Protocol: HTTP | Port: **8000**
- Health check path: `/health`
- Healthy threshold: 2 | Interval: 30s
- Registrar ambas instancias EC2

**Load Balancer:**
- Scheme: Internet-facing
- Listener: HTTP:80 → Target Group
- Subnets: seleccionar subnets **públicas** de us-east-2a y us-east-2b
- Security Group: `pneumonia-lb-sg`

### Verificar el despliegue

```bash
# Health check a través del ALB
curl http://pneumonia-alb-2053601714.us-east-2.elb.amazonaws.com/health
# → {"status": "ok"}

# Ver estado de los contenedores en cada instancia
sudo docker ps
```

---

## Base de datos

La tabla `predicciones` se crea automáticamente cuando la app arranca por primera vez. No necesitas ejecutar ningún script SQL.

### Esquema

```sql
CREATE TABLE predicciones (
    id                   SERIAL PRIMARY KEY,
    imagen_nombre        VARCHAR(255) NOT NULL,   -- UUID + extensión del archivo subido
    resultado            VARCHAR(50)  NOT NULL,   -- "Normal" o "Neumonía"
    porcentaje_confianza NUMERIC(6,2) NOT NULL,   -- Ej: 94.27
    fecha_hora           TIMESTAMP    NOT NULL DEFAULT NOW()
);
```

### Mapeo de clases

El modelo clasifica internamente en 3 clases, que la app simplifica para la base de datos:

| Clase interna | Guardado en DB | Descripción |
|---|---|---|
| `bacteriana` | `Neumonía` | Infiltrado bacteriano visible |
| `viral` | `Neumonía` | Infiltrado viral visible |
| `normal` | `Normal` | Sin signos de neumonía |

### Consultas útiles

```bash
# Ver los últimos 10 diagnósticos
docker exec -it neumonia_db psql -U neumonia_user -d neumonia \
  -c "SELECT * FROM predicciones ORDER BY fecha_hora DESC LIMIT 10;"

# Resumen estadístico
docker exec -it neumonia_db psql -U neumonia_user -d neumonia \
  -c "SELECT resultado, COUNT(*), ROUND(AVG(porcentaje_confianza),2) AS confianza_promedio
      FROM predicciones GROUP BY resultado;"

# Predicciones de las últimas 24 horas
docker exec -it neumonia_db psql -U neumonia_user -d neumonia \
  -c "SELECT * FROM predicciones WHERE fecha_hora > NOW() - INTERVAL '24 hours';"

# Backup de la base de datos
docker exec neumonia_db pg_dump -U neumonia_user neumonia > backup_$(date +%Y%m%d).sql
```

---

## API y endpoints

| Método | Ruta | Descripción | Respuesta |
|---|---|---|---|
| `GET` | `/` | Interfaz web principal | HTML |
| `POST` | `/` | Enviar imagen para análisis | HTML con resultado |
| `GET` | `/health` | Health check para el ALB | `{"status": "ok"}` HTTP 200 |
| `GET` | `/export-pdf` | Descargar reporte PDF | Archivo PDF |
| `GET` | `/export-csv` | Descargar resultado CSV | Archivo CSV |

### Parámetros del POST `/`

| Campo | Tipo | Descripción |
|---|---|---|
| `image` | File | Radiografía (DICOM, JPEG, PNG) |
| `patient_name` | String | Nombre completo del paciente |
| `patient_id` | String | Cédula o ID del paciente |

### Parámetros del GET `/export-pdf`

| Parámetro | Descripción |
|---|---|
| `patient_id` | Cédula del paciente |
| `patient_name` | Nombre del paciente |
| `label` | Diagnóstico (`bacteriana` / `normal` / `viral`) |
| `probability` | Porcentaje de confianza |
| `image` | Ruta relativa a la imagen subida |
| `heatmap` | Ruta relativa al mapa de calor generado |

---

## Pruebas de alta disponibilidad

### Prueba 1 — Failover: una instancia cae

```bash
ALB="http://pneumonia-alb-2053601714.us-east-2.elb.amazonaws.com"

# 1. Verificar que el sistema responde
curl $ALB/health
# → {"status": "ok"}

# 2. Detener la instancia 1 desde AWS Console:
#    EC2 → Instancias → pneumonia-app-1 → Estado de la instancia → Detener

# 3. Esperar ~30 segundos (tiempo del health check del ALB)
# 4. Verificar que el sistema SIGUE respondiendo (instancia 2 absorbe el tráfico)
curl $ALB/health
# → {"status": "ok"}  ✓ Alta disponibilidad confirmada

# 5. Restaurar: EC2 → pneumonia-app-1 → Iniciar
```

### Prueba 2 — Distribución de carga

```bash
ALB="http://pneumonia-alb-2053601714.us-east-2.elb.amazonaws.com"

for i in $(seq 1 20); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" $ALB/health)
  echo "Request $i → HTTP $STATUS"
done
```

### Prueba 3 — Base de datos compartida entre instancias

Sube radiografías desde la app varias veces (el ALB las distribuirá entre las dos instancias) y verifica que todos los registros están centralizados en la DB de la instancia 1:

```bash
# Conectar a instancia 1 y consultar
docker exec -it neumonia_db psql -U neumonia_user -d neumonia \
  -c "SELECT COUNT(*) AS total_predicciones, MIN(fecha_hora), MAX(fecha_hora) FROM predicciones;"
```

Si el `total` incrementa sin importar qué instancia atiende la petición, la arquitectura está funcionando correctamente.

---

## Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `DB_HOST` | Host de PostgreSQL | `db` (instancia 1) / IP privada (instancia 2) |
| `DB_PORT` | Puerto de PostgreSQL | `5432` |
| `DB_NAME` | Nombre de la base de datos | `neumonia` |
| `DB_USER` | Usuario de PostgreSQL | `neumonia_user` |
| `DB_PASSWORD` | Contraseña de PostgreSQL | `changeme` |
| `POSTGRES_DB` | Nombre DB (servicio postgres) | `neumonia` |
| `POSTGRES_USER` | Usuario (servicio postgres) | `neumonia_user` |
| `POSTGRES_PASSWORD` | Contraseña (servicio postgres) | `changeme` |

> **Importante:** Cambia `DB_PASSWORD` y `POSTGRES_PASSWORD` antes de desplegar en producción. Nunca subas contraseñas reales al repositorio.

---

## Stack tecnológico

| Categoría | Tecnología | Uso |
|---|---|---|
| **IA / ML** | TensorFlow 2.20, Keras 3.13 | Inferencia de la CNN y Grad-CAM |
| **Visión computacional** | OpenCV (cv2) | Preprocesamiento CLAHE, conversiones de color |
| **Imágenes médicas** | pydicom | Lectura de archivos DICOM |
| **Web** | Flask 3.0, Gunicorn | Servidor web de producción |
| **Base de datos** | PostgreSQL 15, psycopg2 | Persistencia de predicciones |
| **PDF** | ReportLab | Generación de reportes clínicos |
| **Contenedores** | Docker, Docker Compose | Empaquetado y orquestación |
| **Nube** | AWS EC2, ALB, VPC, S3 | Infraestructura de producción |
| **Lenguaje** | Python 3.12 | Todo el backend |

---

## Comandos de operación

```bash
# Ver logs en tiempo real
sudo docker logs -f neumonia_app
sudo docker logs -f neumonia_db

# Estado de los contenedores
sudo docker ps

# Reiniciar solo la app (sin bajar la DB)
cd /home/appuser/pneumoscan && sudo -u appuser docker compose restart app

# Actualizar la app a la última versión de la imagen
sudo docker pull lamarck558/pneumoscan:latest
cd /home/appuser/pneumoscan && sudo -u appuser docker compose up -d --no-deps app

# Acceder a la consola de PostgreSQL
docker exec -it neumonia_db psql -U neumonia_user -d neumonia

# Backup de la base de datos
docker exec neumonia_db pg_dump -U neumonia_user neumonia > backup_$(date +%Y%m%d).sql
```

---

<div align="center">

**⚠ Aviso médico**

Este sistema es una herramienta de apoyo al diagnóstico basada en inteligencia artificial.
No reemplaza el criterio clínico de un médico especialista.
Cualquier decisión médica debe ser tomada por un profesional de la salud calificado.

---

Desarrollado como parte de la asignatura **Computación en la Nube**
Universidad Autónoma de Occidente

</div>
