import numpy as np
import cv2
from preprocesamiento import preprocesar, TAM, RAIZ, EXT

TAMANOS = [2, 4, 8, 16, 32, 64]   # tamaños de caja (todos dividen a 256)


def recortar_y_normalizar(img, tam=TAM):
    """Recorta al contorno, centra en un cuadro y escala a tam x tam."""
    pts = cv2.findNonZero(img)
    if pts is None:
        return img
    x, y, w, h = cv2.boundingRect(pts)
    rec = img[y:y + h, x:x + w]
    lado = max(w, h)
    cuadro = np.zeros((lado, lado), dtype=np.uint8)
    y0, x0 = (lado - h) // 2, (lado - w) // 2
    cuadro[y0:y0 + h, x0:x0 + w] = rec
    return cv2.resize(cuadro, (tam, tam), interpolation=cv2.INTER_NEAREST)


def contar_cajas(img, s):
    """Cuenta cuántas cajas de s x s contienen al menos un píxel blanco."""
    n = img.shape[0] // s
    bloques = (img[:n * s, :n * s] > 0).reshape(n, s, n, s)
    return int(bloques.any(axis=(1, 3)).sum())


def box_counting(img, tamanos=TAMANOS):
    """Devuelve la dimensión fractal D y los conteos N(s)."""
    N = [contar_cajas(img, s) for s in tamanos]
    x = np.log(1.0 / np.array(tamanos))
    y = np.log(np.array(N, dtype=float))
    pendiente, _ = np.polyfit(x, y, 1)
    return pendiente, N


if __name__ == "__main__":
    # Prueba: 3 imágenes por clase, con bordes y con silueta
    for carpeta in sorted(p for p in RAIZ.iterdir() if p.is_dir()):
        archivos = [f for f in sorted(carpeta.iterdir())
                    if f.suffix.lower() in EXT][:3]
        for f in archivos:
            _, binaria, bordes = preprocesar(f)
            d_bordes, N = box_counting(recortar_y_normalizar(bordes))
            d_silueta, _ = box_counting(recortar_y_normalizar(binaria))
            print(f"{carpeta.name:15s} {f.name:20s} "
                  f"bordes D={d_bordes:.3f}  silueta D={d_silueta:.3f}  N={N}")