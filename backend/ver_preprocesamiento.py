"""Vista previa del preprocesamiento V3 sin entrenar.

Uso:
    py ver_preprocesamiento.py "DATA-SET\\Adidas_Campus\\foto_001.jpg"
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from features_v3 import extract_v3, read_bgr


def fit_canvas(img, size=256):
    h, w = img.shape[:2]
    scale = (size - 20) / max(h, w)
    resized = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.full((size, size, 3), 240, np.uint8)
    h, w = resized.shape[:2]
    canvas[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Compara la imagen con su preprocesamiento V3.')
    parser.add_argument('imagen', type=Path)
    parser.add_argument('--salida', type=Path, default=Path('vista_preprocesamiento.jpg'))
    args = parser.parse_args()
    obj, vector = extract_v3(args.imagen)
    original = fit_canvas(read_bgr(args.imagen))
    normalized = obj.normalized_bgr
    mask = cv2.cvtColor(obj.foreground_mask, cv2.COLOR_GRAY2BGR)
    preview = np.concatenate([original, normalized, mask], axis=1)
    cv2.putText(preview, 'Imagen original', (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (18,18,18), 2)
    cv2.putText(preview, 'Para el modelo', (270, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (18,18,18), 2)
    cv2.putText(preview, 'Mascara', (524, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (18,18,18), 2)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.salida), preview):
        raise OSError(f'No se pudo guardar {args.salida}')
    print('Modo:', obj.processing_mode)
    print('Caracteristicas:', vector.shape[1])
    print('Vista previa guardada:', args.salida.resolve())


if __name__ == '__main__':
    main()
