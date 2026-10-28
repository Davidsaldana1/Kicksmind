from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

RAIZ = Path(r"C:\Users\luise\OneDrive\Escritorio\DATA-SET")
EXT = {".jpg", ".jpeg", ".png", ".webp"}
TAM = 256
MARGEN = 12


def cargar_imagen(ruta):
    # np.fromfile evita errores de OpenCV con rutas con acentos o ñ en Windows
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def preprocesar(ruta):
    img = cargar_imagen(ruta)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, (TAM, TAM))
    suave = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu: calcula solo el umbral que separa tenis y fondo
    _, binaria = cv2.threshold(suave, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Si el marco quedó blanco, el fondo es lo blanco: invertir
    # para que el tenis siempre quede blanco sobre fondo negro
    m = MARGEN
    marco = np.concatenate([binaria[:m].ravel(), binaria[-m:].ravel(),
                            binaria[:, :m].ravel(), binaria[:, -m:].ravel()])
    if (marco > 0).mean() > 0.5:
        binaria = cv2.bitwise_not(binaria)

    bordes = cv2.Canny(suave, 50, 150)
    return gris, binaria, bordes


if __name__ == "__main__":
    # Muestra una imagen de cada clase para revisar visualmente
    clases = sorted(p for p in RAIZ.iterdir() if p.is_dir())
    fig, ejes = plt.subplots(len(clases), 3, figsize=(8, 2.6 * len(clases)))

    for fila, carpeta in enumerate(clases):
        archivo = next(f for f in carpeta.iterdir() if f.suffix.lower() in EXT)
        gris, binaria, bordes = preprocesar(archivo)
        for col, (im, titulo) in enumerate([(gris, "Gris"),
                                            (binaria, "Binaria (Otsu)"),
                                            (bordes, "Bordes (Canny)")]):
            ejes[fila, col].imshow(im, cmap="gray")
            ejes[fila, col].axis("off")
            if fila == 0:
                ejes[fila, col].set_title(titulo)
        ejes[fila, 0].text(-10, 128, carpeta.name, rotation=90,
                           va="center", ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig("revision_preprocesamiento.png", dpi=120)
    plt.show()