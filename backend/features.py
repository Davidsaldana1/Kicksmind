"""Extracción de características de KicksMind.

Mantiene el MISMO preprocesamiento, orden y parámetros usados durante
el entrenamiento original: 8 BC + 8100 HOG + 10 LBP = 8118 características.
Las tres técnicas reutilizan una sola carga de imagen.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

SIZE = 256
BOX_SIZES = (2, 4, 8, 16, 32, 64)
EXPECTED_FEATURES = 8118


@dataclass
class ImageFeatures:
    vector: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    edges: np.ndarray
    box_counts: list[int]
    edge_dimension: float
    silhouette_dimension: float


def preprocess(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grises, máscara de silueta y bordes, igual que en el proyecto original."""
    # fromfile + imdecode permite nombres de archivo con acentos en Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo abrir esta imagen. Prueba un JPG, PNG o WEBP válido.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SIZE, SIZE))
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    m = 12
    border = np.concatenate((
        binary[:m].ravel(), binary[-m:].ravel(),
        binary[:, :m].ravel(), binary[:, -m:].ravel(),
    ))
    if (border > 0).mean() > 0.5:
        binary = cv2.bitwise_not(binary)
    edges = cv2.Canny(smooth, 50, 150)
    return gray, binary, edges


def normalize_object(image: np.ndarray) -> np.ndarray:
    """Recorta el objeto y centra su silueta sin cambiar el método entrenado."""
    points = cv2.findNonZero(image)
    if points is None:
        return image
    x, y, width, height = cv2.boundingRect(points)
    cut = image[y:y + height, x:x + width]
    side = max(width, height)
    centered = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    centered[top:top + height, left:left + width] = cut
    return cv2.resize(centered, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def box_counting(image: np.ndarray) -> tuple[float, list[int]]:
    """Calcula dimensión fractal y las seis cantidades N(s) de cajas."""
    counts = []
    for side in BOX_SIZES:
        n = SIZE // side
        boxes = (image[:n * side, :n * side] > 0).reshape(n, side, n, side)
        counts.append(int(boxes.any(axis=(1, 3)).sum()))

    # En imágenes normales counts > 0 y el resultado coincide con el original.
    # Se evita log(0) para que una imagen sin bordes no rompa la interfaz.
    log_sizes = np.log(1.0 / np.asarray(BOX_SIZES))
    log_counts = np.log(np.maximum(counts, 1))
    dimension = float(np.polyfit(log_sizes, log_counts, 1)[0])
    return dimension, counts


def extract_features(path: str | Path) -> ImageFeatures:
    gray, binary, edges = preprocess(path)

    edge_dimension, counts = box_counting(normalize_object(edges))
    silhouette_dimension, _ = box_counting(normalize_object(binary))
    bc_features = np.array([
        edge_dimension, silhouette_dimension, *np.log(np.maximum(counts, 1))
    ])

    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        transform_sqrt=True, feature_vector=True,
    )

    patterns = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_features, _ = np.histogram(
        patterns.ravel(), bins=np.arange(0, 11), range=(0, 10)
    )
    lbp_features = lbp_features.astype(float)
    lbp_features /= lbp_features.sum()

    vector = np.concatenate((bc_features, hog_features, lbp_features)).reshape(1, -1)
    if vector.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"Se esperaban {EXPECTED_FEATURES} características, no {vector.shape[1]}.")
    return ImageFeatures(
        vector=vector, gray=gray, binary=binary, edges=edges,
        box_counts=counts, edge_dimension=edge_dimension,
        silhouette_dimension=silhouette_dimension,
    )
