"""
Integrator (Facade): conecta lectura, preproceso, modelo, Grad-CAM.
Retorna SOLO lo necesario para la UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from src.data.read_img import read_dicom_image, read_image_file
from src.features.preprocess_img import preprocess_image
from src.models.load_model import load_pneumonia_model
from src.visualizations.grad_cam import generate_gradcam

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probability: float
    heatmap: np.ndarray
    original_image: Image.Image


class PneumoniaDetector:
    def __init__(
        self,
        model_path: str = "models/conv_MLP_84.h5",
        layer_name: str = "conv10_thisone",
    ) -> None:
        self.model = load_pneumonia_model(model_path)
        self.layer_name = layer_name

    def predict(self, file_path: str) -> PredictionResult:
        if file_path.lower().endswith(".dcm"):
            rgb, pil = read_dicom_image(file_path)
        else:
            rgb, pil = read_image_file(file_path)

        batch = preprocess_image(rgb)

        preds = self.model(batch, training=False).numpy()[0]
        class_index = int(np.argmax(preds))
        prob = float(np.max(preds)) * 100.0
        label = LABELS.get(class_index, str(class_index))

        heatmap = generate_gradcam(
            model=self.model,
            image_batch=batch,
            original_rgb=rgb,
            layer_name=self.layer_name,
            class_index=class_index,
        )

        return PredictionResult(
            label=label,
            probability=prob,
            heatmap=heatmap,
            original_image=pil,
        )
