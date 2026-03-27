"""
Lectura de imágenes (DICOM/JPG/PNG) y conversión a:
- array numpy para el pipeline
- imagen PIL para mostrar en la UI
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2
import pydicom
from PIL import Image


def read_dicom_image(file_path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee una imagen DICOM y retorna un array RGB (numpy) y una imagen PIL.

    Args:
        file_path: Ruta al archivo .dcm

    Returns:
        img_rgb: Imagen en RGB (H, W, 3) uint8
        img_pil: Imagen PIL para visualización
    """
    ds = pydicom.dcmread(file_path)
    img_array = ds.pixel_array.astype(np.float32)

    # Normalizar a 0-255 para visualización
    max_val = float(img_array.max()) if img_array.size else 0.0
    img_norm = (np.maximum(img_array, 0) / max_val) * 255.0 if max_val > 0 else img_array
    img_u8 = img_norm.astype(np.uint8)

    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_u8)

    return img_rgb, img_pil


def read_image_file(file_path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee una imagen genérica (jpg/jpeg/png) y retorna array y PIL.

    Args:
        file_path: Ruta al archivo

    Returns:
        img_rgb: Imagen en RGB (H, W, 3) uint8
        img_pil: Imagen PIL
    """
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {file_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    return img_rgb, img_pil
