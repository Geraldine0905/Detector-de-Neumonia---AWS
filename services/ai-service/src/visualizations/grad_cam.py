"""
Grad-CAM TF2 para modelo multiclase.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam(
    model,
    image_batch: np.ndarray,
    original_rgb: np.ndarray,
    layer_name: str = "conv10_thisone",
    class_index: Optional[int] = None,
    threshold: float = 0.10,
) -> np.ndarray:
    """
    Genera heatmap Grad-CAM superpuesto (estilo JET clásico).

    Args:
        model: Modelo Keras.
        image_batch: (1,512,512,1) normalizado.
        original_rgb: (H,W,3) uint8 en RGB.
        layer_name: Capa conv objetivo.
        class_index: índice de clase objetivo (si None, usa argmax).
        threshold: umbral para eliminar ruido en el CAM (0-1).

    Returns:
        np.ndarray: RGB (512,512,3) uint8.
    """
    # Predicción
    preds = model(image_batch, training=False)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = preds.numpy()[0]

    # Si no llega class_index, usamos argmax (esto es clave)
    if class_index is None:
        class_index = int(np.argmax(preds))

    # Modelo para Grad-CAM
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, out = grad_model(image_batch, training=False)

        if isinstance(out, (list, tuple)):
            out = out[0]

        # out shape: (1, num_classes)
        loss = out[:, class_index]

    grads = tape.gradient(loss, conv_out)               # (1,H,W,C)
    weights = tf.reduce_mean(grads, axis=(1, 2))        # (1,C)

    conv_out = conv_out[0].numpy()                      # (H,W,C)
    weights = weights[0].numpy()                        # (C,)

    cam = np.tensordot(conv_out, weights, axes=([2], [0])).astype(np.float32)  # (H,W)

    # ReLU + normalización a [0,1]
    cam = np.maximum(cam, 0)
    cam_max = float(cam.max())
    cam = cam / cam_max if cam_max > 0 else cam

    # Quitar ruido
    if threshold is not None and threshold > 0:
        cam[cam < threshold] = 0

    cam = cv2.resize(cam, (512, 512))

    # Heatmap en BGR (OpenCV)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Base: aseguramos que esté en BGR para mezclar correctamente con OpenCV
    base_rgb = cv2.resize(original_rgb, (512, 512))
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    # Overlay clásico
    superimposed_bgr = cv2.addWeighted(base_bgr, 0.6, heatmap_bgr, 0.4, 0)

    # Volver a RGB para PIL / UI
    superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)

    return superimposed_rgb.astype(np.uint8)
