"""
Frontend service — PneumoScan
Serves the web UI and coordinates with ai-service via HTTP.
No TensorFlow dependency.
"""

import os
import csv
import uuid
import io
from pathlib import Path
from datetime import datetime

import requests
from flask import (
    Flask, render_template, request, Response,
    send_file, jsonify, send_from_directory,
)
from PIL import Image
import psycopg2
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)

# ── Service URL (resolved via Kubernetes DNS or docker-compose alias) ──
AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://ai-service:8000")

# ── File storage (ephemeral — use EFS/S3 in production) ──
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/uploads")
HEATMAP_FOLDER = os.environ.get("HEATMAP_FOLDER", "/tmp/heatmaps")
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(HEATMAP_FOLDER).mkdir(parents=True, exist_ok=True)

# ── Label mapping: internal → clinical ──
RESULTADO_MAP = {"normal": "Normal", "bacteriana": "Neumonía", "viral": "Neumonía"}


# ────────────────────────────────────────────────────────────────────
# Database helpers
# ────────────────────────────────────────────────────────────────────

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "db-service"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "neumonia"),
        user=os.environ.get("DB_USER", "neumonia_user"),
        password=os.environ.get("DB_PASSWORD", "changeme"),
    )


def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predicciones (
                id                   SERIAL PRIMARY KEY,
                imagen_nombre        VARCHAR(255) NOT NULL,
                resultado            VARCHAR(50)  NOT NULL,
                porcentaje_confianza NUMERIC(6,2) NOT NULL,
                fecha_hora           TIMESTAMP    NOT NULL DEFAULT NOW()
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("[DB] Table 'predicciones' ready.")
    except Exception as e:
        print(f"[DB] Warning: couldn't initialize database: {e}")


def save_prediction(imagen_nombre: str, resultado: str, confianza: float):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predicciones (imagen_nombre, resultado, porcentaje_confianza, fecha_hora) "
            "VALUES (%s, %s, %s, %s)",
            (imagen_nombre, resultado, round(confianza, 2), datetime.now()),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB] Error saving prediction: {e}")


init_db()


# ────────────────────────────────────────────────────────────────────
# Static file routes for dynamically generated files
# ────────────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/heatmaps/<path:filename>")
def heatmap_file(filename):
    return send_from_directory(HEATMAP_FOLDER, filename)


# ────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "frontend"}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html")

        ext = Path(file.filename).suffix.lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(upload_path)

        # ── Call ai-service ──
        try:
            with open(upload_path, "rb") as f:
                resp = requests.post(
                    f"{AI_SERVICE_URL}/predict",
                    files={"image": (file.filename, f, "application/octet-stream")},
                    timeout=120,
                )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"[AI-SERVICE] Error: {e}")
            return render_template("index.html", error=str(e))

        label = result["label"]
        prob_value = float(result["probability"])
        heatmap_b64 = result["heatmap_base64"]
        prob_class = "danger" if prob_value > 70 else "ok"

        # ── Decode and save heatmap ──
        import base64
        heatmap_name = f"heatmap_{unique_name.replace(ext, '.png')}"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_name)
        heatmap_bytes = base64.b64decode(heatmap_b64)
        with open(heatmap_path, "wb") as fh:
            fh.write(heatmap_bytes)

        # ── Persist to DB ──
        resultado_clinico = RESULTADO_MAP.get(label, label)
        save_prediction(unique_name, resultado_clinico, prob_value)

        patient_id   = request.form.get("patient_id", "")
        patient_name = request.form.get("patient_name", "")

        return render_template(
            "index.html",
            label=label,
            probability=f"{prob_value:.2f}",
            prob_class=prob_class,
            image=unique_name,
            heatmap_image=heatmap_name,
            patient_id=patient_id,
            patient_name=patient_name,
        )

    return render_template("index.html")


@app.route("/export-pdf")
def export_pdf():
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
        topMargin=2*cm, bottomMargin=2*cm,
    )

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
    W = A4[0] - 4*cm

    story.append(Paragraph("PneumoScan", style_title))
    story.append(Paragraph("Medical Diagnostic Support System for Pneumonia", style_subtitle))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#005cff")))
    story.append(Spacer(1, 0.4*cm))

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    meta_data = [["Report date:", now], ["Folio:", uuid.uuid4().hex[:8].upper()]]
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

    story.append(Paragraph("Patient Data", style_section))
    pt = Table([["Full name:", patient_name], ["ID / Passport:", patient_id]], colWidths=[4.5*cm, W-4.5*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#1a1a2e")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f4f6fb"), colors.white]),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Analysis Result", style_section))
    diag_color = colors.HexColor("#ff4d6d") if label != "Normal" else colors.HexColor("#00c9a7")
    rt = Table([["Diagnosis:", label], ["Model confidence:", f"{probability}%"]], colWidths=[5.5*cm, W-5.5*cm])
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

    img_w = (W - 0.8*cm) / 2
    img_h = img_w * 0.9

    def load_rl_image(folder, filename, w, h):
        full = os.path.join(folder, filename)
        if filename and os.path.exists(full):
            return RLImage(full, width=w, height=h)
        placeholder = io.BytesIO()
        Image.new("RGB", (300, 270), color=(220, 224, 235)).save(placeholder, "PNG")
        placeholder.seek(0)
        return RLImage(placeholder, width=w, height=h)

    img_original = load_rl_image(UPLOAD_FOLDER, image_file, img_w, img_h)
    img_heatmap  = load_rl_image(HEATMAP_FOLDER, heatmap_file, img_w, img_h)

    img_table = Table(
        [[Paragraph("Original X-Ray", style_body), Paragraph("Grad-CAM Heatmap", style_body)],
         [img_original, img_heatmap]],
        colWidths=[img_w, img_w], hAlign="CENTER",
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#5a6070")),
        ("BOTTOMPADDING", (0,0), (-1,0), 5),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.8*cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "This report is generated by an AI system for diagnostic support purposes. "
        "It does not replace the clinical judgment of a medical specialist.",
        style_disclaimer,
    ))

    doc.build(story)
    buffer.seek(0)

    filename = f"Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, mimetype="application/pdf",
                     as_attachment=True, download_name=filename)


@app.route("/export-csv")
def export_csv():
    patient_id  = request.args.get("patient_id", "")
    label       = request.args.get("label", "")
    probability = request.args.get("probability", "")

    def generate():
        yield "cedula,diagnostico,probabilidad\n"
        yield f"{patient_id},{label},{probability}%\n"

    headers = {
        "Content-Disposition": f"attachment; filename=report_{patient_id or 'patient'}.csv",
        "Content-Type": "text/csv",
    }
    return Response(generate(), headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
