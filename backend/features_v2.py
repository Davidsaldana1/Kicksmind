"""Características mejoradas de KicksMind.

El vector de predicción v2 contiene, en este orden:
  8 Box Counting + 8,100 HOG a 256 px + 10 LBP
  + 8,100 HOG a 128 px + 144 histogramas HSV = 16,362 valores.

Usar EXACTAMENTE esta función tanto en entrenamiento como en la interfaz.
"""
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

from features import ImageFeatures, extract_features

FEATURE_COUNT_V2 = 16362


def extract_v2(path: str | Path, base: ImageFeatures | None = None) -> tuple[ImageFeatures, np.ndarray]:
    """Devuelve características visuales y el vector listo para el modelo v2."""
    base = base if base is not None else extract_features(path)

    # HOG con otra escala: ayuda a ver patrones pequeños y grandes del tenis.
    small_gray = cv2.resize(base.gray, (128, 128), interpolation=cv2.INTER_AREA)
    hog_small = hog(
        small_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    # Histograma HSV en la imagen completa, mitad superior e inferior.
    # Se mantiene el orden usado al entrenar el modelo.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histograms = []
    for region in (hsv, hsv[:128], hsv[128:]):
        hist = cv2.calcHist([region], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        hist = (hist / (hist.sum() + 1e-8)).astype(np.float32)
        histograms.append(hist)
    color = np.concatenate(histograms)

    vector = np.concatenate((base.vector.ravel(), hog_small, color * 0.3))
    if vector.size != FEATURE_COUNT_V2:
        raise ValueError(f"Características esperadas: {FEATURE_COUNT_V2}, obtenidas: {vector.size}")
    return base, vector.reshape(1, -1)
