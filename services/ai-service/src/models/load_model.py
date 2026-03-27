"""
Carga del modelo Keras.
"""

from __future__ import annotations

import os; from tensorflow.keras.models import load_model # type: ignore


def load_pneumonia_model(model_path: str = "models/conv_MLP_84.h5"):
    """
    Carga el modelo desde un archivo .h5.

    Args:
        model_path: Ruta del modelo.

    Returns:
        Modelo Keras.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

    return load_model(model_path, compile=False)

