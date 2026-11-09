
import numpy as np
from skimage.feature import hog, local_binary_pattern

from preprocesamiento import preprocesar, RAIZ, EXT


# CONFIGURACIÓN HOG
ORIENTACIONES = 9
PIXELES_CELDA = (16, 16)
CELDAS_BLOQUE = (2, 2)

# CONFIGURACIÓN LBP
PUNTOS = 8
RADIO = 1
METODO = "uniform"


def extraer_hog(ruta):
    # Obtenemos la imagen en escala de grises
    gris, _, _ = preprocesar(ruta)

    # Extraemos las características HOG
    caracteristicas = hog(
        gris,
        orientations=ORIENTACIONES,
        pixels_per_cell=PIXELES_CELDA,
        cells_per_block=CELDAS_BLOQUE,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return caracteristicas


def extraer_lbp(ruta):
    # Obtenemos la imagen en escala de grises
    gris, _, _ = preprocesar(ruta)

    # Calculamos los patrones LBP
    patrones = local_binary_pattern(
        gris,
        PUNTOS,
        RADIO,
        method=METODO
    )

    # Creamos un histograma de frecuencias
    histograma, _ = np.histogram(
        patrones.ravel(),
        bins=np.arange(0, PUNTOS + 3),
        range=(0, PUNTOS + 2)
    )

    # Normalizamos el histograma
    histograma = histograma.astype(float)
    histograma /= histograma.sum()

    return histograma


if __name__ == "__main__":

    print("===== PRUEBA DE HOG Y LBP =====")

    clases = sorted(
        carpeta for carpeta in RAIZ.iterdir()
        if carpeta.is_dir()
    )

    for carpeta in clases:

        imagenes = [
            archivo for archivo in carpeta.iterdir()
            if archivo.suffix.lower() in EXT
        ]

        if not imagenes:
            print(f"Sin imágenes: {carpeta.name}")
            continue

        ruta = imagenes[0]

        caracteristicas_hog = extraer_hog(ruta)
        caracteristicas_lbp = extraer_lbp(ruta)

        print(f"\nClase: {carpeta.name}")
        print("Características HOG:", len(caracteristicas_hog))
        print("Características LBP:", len(caracteristicas_lbp))

    print("\n¡Prueba finalizada!")
