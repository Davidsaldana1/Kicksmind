"""KicksMind V3 experimental: aislamiento conservador + GrabCut en fondos complejos.

La función `extract_v3` DEBE usarse igual al entrenar y al predecir.
Aísla primero objetos en fondos uniformes, luego prueba una máscara
alternativa para fondos planos con poco contraste. Con fondos complejos
intenta GrabCut y valida el resultado antes de usarlo. Si no hay evidencia
suficiente, mantiene la foto completa. No promete segmentación perfecta.

Vector: 8 BC + 8 100 HOG 256 + 10 LBP + 8 100 HOG 128 + 144 HSV.
El total coincide numéricamente con V2 (16 362), pero sus valores NO
son compatibles con un modelo V2 previamente entrenado.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from features import ImageFeatures, box_counting, normalize_object

SIZE = 256
FEATURE_COUNT_V3 = 16362


@dataclass
class V3ImageFeatures(ImageFeatures):
    processing_mode: str = ""
    normalized_bgr: np.ndarray | None = None
    foreground_mask: np.ndarray | None = None


def read_bgr(path: str | Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}. Conviértela a JPG o PNG.")
    return img


def detect_uniform_background(image: np.ndarray) -> np.ndarray | None:
    """Máscara de objeto solamente si el borde del fondo es bastante uniforme.

    Es conservadora: si no hay evidencia clara, vuelve None para no recortar
    un sneaker parcialmente ni entrenar con una silueta inventada.
    """
    h, w = image.shape[:2]
    if min(h, w) < 60:
        return None
    # Analizar miniatura evita consumir mucha RAM con fotos de varios megapixeles.
    if max(h, w) > 900:
        scale = 900 / max(h, w)
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        found = detect_uniform_background(small)
        return None if found is None else cv2.resize(found, (w, h), interpolation=cv2.INTER_NEAREST)
    sample = max(3, min(h, w) // 24)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    rim = np.concatenate([
        lab[:sample].reshape(-1, 3), lab[-sample:].reshape(-1, 3),
        lab[:, :sample].reshape(-1, 3), lab[:, -sample:].reshape(-1, 3),
    ])
    background = np.median(rim, axis=0)
    rim_d = np.linalg.norm(rim - background, axis=1)
    if np.percentile(rim_d, 85) > 21:
        return None
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > 27).astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(h, w) / 125)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None
    component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    area = int(stats[component, cv2.CC_STAT_AREA])
    fraction = area / (w * h)
    x, y, bw, bh, _ = stats[component]
    if not (0.035 < fraction < 0.78 and bw >= w * .25 and bh >= h * .17):
        return None
    # Evita aislar trozos diminutos, fondos que invaden el cuadro o cortes extremos.
    if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
        return None
    foreground = np.where(labels == component, 255, 0).astype(np.uint8)
    # Convex hull rellena huecos: un tenis tiene partes claras y oscuras.
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        foreground = cv2.drawContours(np.zeros_like(foreground),
                                      [cv2.convexHull(max(contours, key=cv2.contourArea))],
                                      -1, 255, cv2.FILLED)
    return foreground




def detect_soft_uniform_background(image: np.ndarray) -> np.ndarray | None:
    """Segundo intento prudente para fondos casi planos y tenis de color parecido.

    No modifica el modelo ni presupone que el tenis esté completo. Rechaza la
    segmentación cuando el fondo no es homogéneo o el objeto toca los bordes.
    """
    height, width = image.shape[:2]
    if min(height, width) < 90:
        return None

    # Trabajar con una copia pequeña reduce memoria y tiempo de procesamiento.
    scale = min(1.0, 900.0 / max(height, width))
    if scale < 1:
        small = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image
    h, w = small.shape[:2]

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(3, min(h, w) // 30)
    rim = np.concatenate([
        lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
        lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3),
    ])
    background = np.median(rim, axis=0)
    rim_distance = np.linalg.norm(rim - background, axis=1)

    # Solo acepta fondos homogéneos. Un borde con texturas, piel, mesas o
    # parte de un zapato recortado debe seguir el comportamiento anterior.
    if np.percentile(rim_distance, 90) > 13.0:
        return None

    distance = np.linalg.norm(lab - background, axis=2)
    threshold = max(12.0, float(np.percentile(rim_distance, 98)) + 7.0)
    candidate = (distance > threshold).astype(np.uint8)

    # Cerrar pequeñas discontinuidades sin romper regiones claras del zapato.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if n < 2:
        return None
    component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    x, y, bw, bh, area = map(int, stats[component])
    fraction = area / float(w * h)
    if not (0.055 <= fraction <= 0.70 and bw >= w * 0.28 and bh >= h * 0.17):
        return None

    # Evitar cortar zapatillas que salen del encuadre, como foto_031 de AF1.
    border_x = max(2, round(w * 0.012))
    border_y = max(2, round(h * 0.012))
    if x <= border_x or y <= border_y or x + bw >= w - border_x or y + bh >= h - border_y:
        return None

    component_mask = np.where(labels == component, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    hull_area = cv2.contourArea(hull)
    # Un casco convexo desproporcionado suele representar regiones separadas
    # o una sombra, no una silueta fiable.
    if hull_area > area * 1.60:
        return None
    mask = cv2.drawContours(np.zeros((h, w), dtype=np.uint8), [hull], -1, 255, cv2.FILLED)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def detect_complex_background(image: np.ndarray) -> np.ndarray | None:
    """Intenta aislar el zapato en fondos complejos mediante GrabCut.

    Solo se acepta una mascara si supera controles geometricos y de contraste.
    De otro modo no se recorta la foto (sin suposiciones peligrosas).
    """
    height, width = image.shape[:2]
    if min(height, width) < 90:
        return None
    factor = min(1.0, 380 / max(height, width))
    if factor < 1:
        small = cv2.resize(image, (max(1, int(width * factor)), max(1, int(height * factor))), interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()
    h, w = small.shape[:2]
    margin_x, margin_y = max(3, int(w * .055)), max(3, int(h * .055))
    rectangle = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    if rectangle[2] < 50 or rectangle[3] < 50:
        return None
    bg = np.zeros((1,65), np.float64)
    fg = np.zeros((1,65), np.float64)
    gc_mask = np.zeros((h,w),np.uint8)
    try:
        cv2.grabCut(small, gc_mask, rectangle, bg, fg, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    possible = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
    if not possible.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    possible = cv2.morphologyEx(possible,cv2.MORPH_CLOSE,k)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(possible,8)
    if count < 2:
        return None
    label = int(np.argmax(stats[1:,cv2.CC_STAT_AREA]) + 1)
    area = int(stats[label,cv2.CC_STAT_AREA])
    x,y,bw,bh,_ = stats[label]
    frac = area / (h*w)
    # Conservar solo objetos individuales y bien delimitados.
    if not (.055 <= frac <= .72 and bw >= w*.28 and bh >= h*.17):
        return None
    if area / max(1, int(possible.sum())) < .75:
        return None
    if x < margin_x or y < margin_y or x+bw > w-margin_x or y+bh > h-margin_y:
        return None
    # Asegurar que el objeto tiene suficiente contraste frente al fondo estimado.
    comp = (labels==label).astype(np.uint8)
    inner = cv2.erode(comp, k, iterations=2) > 0
    rim = np.zeros((h,w),np.uint8)
    rim[:margin_y,:] = 1;rim[-margin_y:,:] = 1
    rim[:,:margin_x] = 1;rim[:,-margin_x:] = 1
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB).astype(np.float32)
    fg_col=np.median(lab[inner] if inner.any() else lab[comp>0],axis=0)
    bg_col=np.median(lab[rim>0],axis=0)
    if float(np.linalg.norm(fg_col-bg_col)) < 18:
        return None
    contours,_ = cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull=cv2.convexHull(max(contours,key=cv2.contourArea))
    comp=cv2.drawContours(np.zeros_like(comp),[hull],-1,1,cv2.FILLED)
    if (h,w) != (height,width):
        comp=cv2.resize(comp,(width,height),interpolation=cv2.INTER_NEAREST)
    return comp.astype(np.uint8)*255

def letterbox_image(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Recorta solo si hay máscara fiable. Conserva proporciones y centra a 256 px."""
    h, w = image.shape[:2]
    if mask is not None:
        yy, xx = np.where(mask > 0)
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        padx = max(4, int((x1 - x0) * .08))
        pady = max(4, int((y1 - y0) * .08))
        x0, x1 = max(0, x0 - padx), min(w, x1 + padx)
        y0, y1 = max(0, y0 - pady), min(h, y1 + pady)
        image = image[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
    else:
        mask = np.full((h, w), 255, dtype=np.uint8)
    hh, ww = image.shape[:2]
    scale = 232.0 / max(hh, ww)
    new_w = max(1, min(SIZE, int(round(ww * scale))))
    new_h = max(1, min(SIZE, int(round(hh * scale))))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # Un fondo neutro constante reduce cambios de brillo de los originales limpios.
    canvas = np.full((SIZE, SIZE, 3), 240, dtype=np.uint8)
    region_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    y0, x0 = (SIZE - new_h) // 2, (SIZE - new_w) // 2
    # Usar máscara solo si el fondo fue aislado; evita difuminar el borde del zapato.
    if (resized_mask == 255).all():
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    else:
        roi = canvas[y0:y0 + new_h, x0:x0 + new_w]
        roi[resized_mask > 0] = resized[resized_mask > 0]
    region_mask[y0:y0 + new_h, x0:x0 + new_w] = resized_mask
    return canvas, region_mask


def extract_v3(path: str | Path) -> tuple[V3ImageFeatures, np.ndarray]:
    image = read_bgr(path)
    object_mask = detect_uniform_background(image)
    if object_mask is not None:
        mode = "Fondo uniforme: objeto aislado"
    else:
        # Nuevo: fondo gris/beige casi uniforme, aunque contraste poco.
        object_mask = detect_soft_uniform_background(image)
        if object_mask is not None:
            mode = "Fondo uniforme suave: objeto aislado"
        else:
            object_mask = detect_complex_background(image)
            mode = ("Fondo complejo: objeto aislado con GrabCut"
                    if object_mask is not None else "Foto completa: sin recorte del objeto")
    normalized, mask = letterbox_image(image, object_mask)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    # Para BC usar la silueta segmentada, cuando existe; en caso contrario,
    # una aproximación por Otsu, como en V2.
    if object_mask is not None:
        binary = mask.copy()
    else:
        _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        border_pixels = np.concatenate([binary[:12].ravel(), binary[-12:].ravel(),
                                        binary[:, :12].ravel(), binary[:, -12:].ravel()])
        if (border_pixels > 0).mean() > .5:
            binary = cv2.bitwise_not(binary)
    edges = cv2.Canny(smooth, 50, 150)
    if object_mask is not None:
        edges = cv2.bitwise_and(edges, mask)

    edge_dimension, counts = box_counting(normalize_object(edges))
    silhouette_dimension, _ = box_counting(normalize_object(binary))
    bc = np.array([edge_dimension, silhouette_dimension, *np.log(np.maximum(counts, 1))], dtype=np.float32)

    hog_large = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    small_gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    hog_small = hog(small_gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    patterns = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp, _ = np.histogram(patterns.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp = lbp.astype(np.float32) / max(lbp.sum(), 1)

    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    histograms = []
    for start, end in [(0, 256), (0, 128), (128, 256)]:
        roi = hsv[start:end]
        region_mask = mask[start:end]
        hist = cv2.calcHist([roi], [0, 1], region_mask, [12, 4], [0, 180, 0, 256]).ravel()
        histograms.append((hist / max(hist.sum(), 1e-8)).astype(np.float32))
    color = np.concatenate(histograms)
    vector = np.concatenate([bc, hog_large, lbp, hog_small, color * .3]).astype(np.float32)
    if vector.size != FEATURE_COUNT_V3:
        raise AssertionError(f"V3 esperaba {FEATURE_COUNT_V3} características: {vector.size}")
    result = V3ImageFeatures(
        vector=vector.reshape(1, -1), gray=gray, binary=binary, edges=edges,
        box_counts=counts, edge_dimension=edge_dimension,
        silhouette_dimension=silhouette_dimension, processing_mode=mode,
        normalized_bgr=normalized, foreground_mask=mask)
    return result, vector.reshape(1, -1)
