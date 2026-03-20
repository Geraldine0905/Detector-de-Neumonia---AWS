import sys
import os
import csv
import uuid
import io
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, redirect, url_for, Response, send_file, jsonify
from PIL import Image
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from src.app.integrator import PneumoniaDetector

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "ui/static/uploads"
app.config["HEATMAP_FOLDER"] = "ui/static/heatmaps"

# Crear carpetas si no existen
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
Path(app.config["HEATMAP_FOLDER"]).mkdir(parents=True, exist_ok=True)

detector = PneumoniaDetector(model_path="models/conv_MLP_84.h5")

# ── Mapeo de etiquetas internas → resultado clínico ──────────────────────────
RESULTADO_MAP = {"normal": "Normal", "bacteriana": "Neumonía", "viral": "Neumonía"}


# ── Conexión a PostgreSQL ─────────────────────────────────────────────────────
def get_db_connection():
    """Devuelve una conexión a PostgreSQL usando variables de entorno."""
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "neumonia"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
    )


def init_db():
    """Crea la tabla de predicciones si no existe."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predicciones (
                id                  SERIAL PRIMARY KEY,
                imagen_nombre       VARCHAR(255) NOT NULL,
                resultado           VARCHAR(50)  NOT NULL,
                porcentaje_confianza NUMERIC(6,2) NOT NULL,
                fecha_hora          TIMESTAMP    NOT NULL DEFAULT NOW()
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("[DB] Tabla 'predicciones' lista.")
    except Exception as e:
        print(f"[DB] Advertencia: no se pudo inicializar la base de datos: {e}")


def save_prediction(imagen_nombre: str, resultado: str, confianza: float):
    """Guarda una predicción en la base de datos."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predicciones (imagen_nombre, resultado, porcentaje_confianza, fecha_hora)
            VALUES (%s, %s, %s, %s)
            """,
            (imagen_nombre, resultado, round(confianza, 2), datetime.now()),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB] Error al guardar predicción: {e}")


# Inicializar DB al arrancar
init_db()


@app.route("/health")
def health():
    """Health check para el Application Load Balancer de AWS."""
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html")

        # Guardar imagen con nombre único para evitar colisiones
        ext = Path(file.filename).suffix.lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        # Inferencia
        result = detector.predict(filepath)

        patient_id   = request.form.get("patient_id", "")
        patient_name = request.form.get("patient_name", "")
        prob_value   = result.probability
        prob_class   = "danger" if prob_value > 70 else "ok"

        # Guardar predicción en la base de datos
        resultado_clinico = RESULTADO_MAP.get(result.label, result.label)
        save_prediction(unique_name, resultado_clinico, prob_value)

        # Guardar heatmap — result.heatmap es numpy array uint8 (H,W,3)
        heatmap_name = f"heatmap_{unique_name.replace(ext, '.png')}"
        heatmap_path = os.path.join(app.config["HEATMAP_FOLDER"], heatmap_name)
        Image.fromarray(result.heatmap.astype("uint8")).save(heatmap_path)

        return render_template(
            "index.html",
            label=result.label,
            probability=f"{prob_value:.2f}",
            prob_class=prob_class,
            image=f"uploads/{unique_name}",
            heatmap_image=f"heatmaps/{heatmap_name}",
            patient_id=patient_id,
            patient_name=patient_name,
        )

    return render_template("index.html")


@app.route("/export-pdf")
def export_pdf():
    """Genera y descarga un PDF con el reporte del diagnóstico."""
    patient_id   = request.args.get("patient_id", "Sin ID")
    patient_name = request.args.get("patient_name", "Sin nombre")
    label        = request.args.get("label", "—")
    probability  = request.args.get("probability", "—")
    image_file   = request.args.get("image", "")
    heatmap_file = request.args.get("heatmap", "")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    # ── estilos ──
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle("title",
        fontName="Helvetica-Bold", fontSize=18,
        textColor=colors.HexColor("#0d1117"),
        spaceAfter=4, alignment=TA_CENTER)
    style_subtitle = ParagraphStyle("subtitle",
        fontName="Helvetica", fontSize=10,
        textColor=colors.HexColor("#5a6070"),
        spaceAfter=2, alignment=TA_CENTER)
    style_section = ParagraphStyle("section",
        fontName="Helvetica-Bold", fontSize=11,
        textColor=colors.HexColor("#005cff"),
        spaceBefore=14, spaceAfter=6)
    style_body = ParagraphStyle("body",
        fontName="Helvetica", fontSize=10,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=4, leading=15)
    style_disclaimer = ParagraphStyle("disclaimer",
        fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER, spaceBefore=10)

    story = []
    W = A4[0] - 4*cm  # ancho útil

    # ── encabezado ──
    story.append(Paragraph("PneumoScan", style_title))
    story.append(Paragraph("Sistema de Apoyo al Diagnóstico Médico de Neumonía", style_subtitle))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#005cff")))
    story.append(Spacer(1, 0.4*cm))

    # ── fecha y folio ──
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    meta_data = [
        ["Fecha del reporte:", now],
        ["Folio:", uuid.uuid4().hex[:8].upper()],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, W-4*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#5a6070")),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # ── datos del paciente ──
    story.append(Paragraph("Datos del Paciente", style_section))
    patient_data = [
        ["Nombre completo:", patient_name],
        ["Cédula / ID:", patient_id],
    ]
    pt = Table(patient_data, colWidths=[4.5*cm, W-4.5*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (1,0), (1,-1), colors.HexColor("#1a1a2e")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f4f6fb"), colors.white]),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.5*cm))

    # ── resultado ──
    story.append(Paragraph("Resultado del Análisis", style_section))

    diag_color = colors.HexColor("#ff4d6d") if label == "PNEUMONIA" else colors.HexColor("#00c9a7")
    result_data = [
        ["Diagnóstico:", label],
        ["Confianza en el modelo:", f"{probability}%"],
    ]
    rt = Table(result_data, colWidths=[5.5*cm, W-5.5*cm])
    rt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 11),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (1,0), (1,-1), diag_color),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f4f6fb"), colors.white]),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (-1,-1), 9),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#e0e4ef")),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.6*cm))

    # ── imágenes ──
    img_w = (W - 0.8*cm) / 2
    img_h = img_w * 0.9

    def load_rl_image(rel_path, w, h):
        full = os.path.join("ui/static", rel_path)
        if rel_path and os.path.exists(full):
            return RLImage(full, width=w, height=h)
        # placeholder gris
        placeholder = io.BytesIO()
        Image.new("RGB", (300, 270), color=(220, 224, 235)).save(placeholder, "PNG")
        placeholder.seek(0)
        return RLImage(placeholder, width=w, height=h)

    img_original = load_rl_image(image_file, img_w, img_h)
    img_heatmap  = load_rl_image(heatmap_file, img_w, img_h)

    img_table = Table(
        [[Paragraph("Radiografía original", style_body),
          Paragraph("Mapa de calor Grad-CAM", style_body)],
         [img_original, img_heatmap]],
        colWidths=[img_w, img_w], hAlign="CENTER"
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#5a6070")),
        ("BOTTOMPADDING", (0,0), (-1,0), 5),
        ("TOPPADDING", (1,0), (-1,-1), 0),
        ("COLPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.8*cm))

    # ── línea final ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "⚠ Este reporte es generado por un sistema de inteligencia artificial con fines de apoyo diagnóstico. "
        "No reemplaza el criterio clínico de un médico especialista.",
        style_disclaimer
    ))

    doc.build(story)
    buffer.seek(0)

    filename = f"Reporte_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, mimetype="application/pdf",
                     as_attachment=True, download_name=filename)


@app.route("/export-csv")
def export_csv():
    """Exporta el resultado actual como descarga CSV."""
    patient_id  = request.args.get("patient_id", "")
    label       = request.args.get("label", "")
    probability = request.args.get("probability", "")

    def generate():
        yield "cedula,diagnostico,probabilidad\n"
        yield f"{patient_id},{label},{probability}%\n"

    headers = {
        "Content-Disposition": f"attachment; filename=reporte_{patient_id or 'paciente'}.csv",
        "Content-Type": "text/csv",
    }
    return Response(generate(), headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)