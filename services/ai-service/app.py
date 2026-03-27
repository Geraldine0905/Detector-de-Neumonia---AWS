"""
AI Service — PneumoScan
REST API that loads the TensorFlow model and returns predictions.
No Flask UI, no database writes, no ReportLab.
"""

import os
import base64
import io
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

from src.app.integrator import PneumoniaDetector

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/conv_MLP_84.h5")

# Load model once at startup — warm-up happens here, not per-request
print(f"[AI-SERVICE] Loading model from {MODEL_PATH} …")
detector = PneumoniaDetector(model_path=MODEL_PATH)
print("[AI-SERVICE] Model ready.")


# ────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Health check for Kubernetes liveness/readiness probes."""
    return jsonify({"status": "ok", "service": "ai-service"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data POST with an 'image' file field.
    Returns JSON:
        {
            "label":           str,    # "bacteriana" | "normal" | "viral"
            "probability":     float,  # 0.0 – 100.0
            "heatmap_base64":  str     # PNG encoded as base64
        }
    """
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No 'image' field in request"}), 400

    # Save to a temp file so the pipeline can read it by path
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = detector.predict(tmp_path)

        # Encode heatmap as base64 PNG
        heatmap_img = Image.fromarray(result.heatmap.astype("uint8"))
        buf = io.BytesIO()
        heatmap_img.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "label":          result.label,
            "probability":    round(result.probability, 2),
            "heatmap_base64": heatmap_b64,
        })

    except Exception as e:
        app.logger.error(f"[PREDICT] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
