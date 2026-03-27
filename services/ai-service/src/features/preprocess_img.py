"""
Preprocesamiento: resize, gray, CLAHE, normalización, batch.
"""

from __future__ import annotations

import numpy as np
import cv2


def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocesa imagen a (1, 512, 512, 1) normalizada.

    Args:
        image_rgb: np.ndarray (H,W,3) o (H,W)

    Returns:
        np.ndarray: batch (1,512,512,1)
    """
    img = cv2.resize(image_rgb, (512, 512))

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray.astype(np.uint8))

    eq = eq.astype(np.float32) / 255.0

    batch = np.expand_dims(eq, axis=-1)
    batch = np.expand_dims(batch, axis=0)

    return batch
