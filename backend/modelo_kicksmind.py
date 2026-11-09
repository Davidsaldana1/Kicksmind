
import numpy as np
import joblib

from box_counting import box_counting, recortar_y_normalizar
from hog_lbp import extraer_hog, extraer_lbp
from preprocesamiento import preprocesar


# ==========================================
# 1. CONFIGURACION
# ==========================================

ARCHIVO_X = "X_combinado.npy"
ARCHIVO_Y = "y.npy"

ARCHIVO_MODELO = "modelo_final_kicksmind.joblib"


# ==========================================
# 2. EXTRAER CARACTERISTICAS BOX COUNTING
# ==========================================

def extraer_caracteristicas_bc(ruta):

    # Preprocesar la imagen
    _, binaria, bordes = preprocesar(ruta)

    # Obtener dimension fractal y conteos de bordes
    d_b, N_b = box_counting(
        recortar_y_normalizar(bordes)
    )

    # Obtener dimension fractal de la silueta
    d_s, _ = box_counting(
        recortar_y_normalizar(binaria)
    )

    # Construir las 8 caracteristicas
    caracteristicas = [d_b, d_s] + list(
        np.log(np.maximum(N_b, 1))
    )

    return np.array(caracteristicas)


# ==========================================
# 3. EXTRAER CARACTERISTICAS COMBINADAS
# ==========================================

def extraer_caracteristicas_combinadas(ruta):

    # Extraer las caracteristicas de cada metodo
    bc = extraer_caracteristicas_bc(ruta)
    hog = extraer_hog(ruta)
    lbp = extraer_lbp(ruta)

    # Combinar en el mismo orden del entrenamiento
    combinadas = np.concatenate([
        bc,
        hog,
        lbp
    ])

    # Convertir a matriz de una sola imagen
    return combinadas.reshape(1, -1)


# ==========================================
# 4. FUNCION PARA PREDECIR UNA IMAGEN
# ==========================================

def predecir_sneaker(ruta_imagen):

    # Cargar el modelo entrenado
    clasificador = joblib.load(ARCHIVO_MODELO)

    # Extraer las caracteristicas de la imagen nueva
    caracteristicas = extraer_caracteristicas_combinadas(
        ruta_imagen
    )

    # Comprobar que las dimensiones coincidan
    if caracteristicas.shape[1] != clasificador.n_features_in_:
        raise ValueError(
            "Las caracteristicas de la imagen no coinciden con el modelo."
        )

    # Realizar la prediccion
    prediccion = clasificador.predict(caracteristicas)[0]

    # Obtener probabilidades por categoria
    probabilidades = clasificador.predict_proba(caracteristicas)[0]

    print("\n===== RESULTADO DE KICKSMIND =====")
    print("Sneaker identificado:", prediccion)

    print("\nProbabilidades por categoria:")

    for clase, probabilidad in zip(clasificador.classes_, probabilidades):
        print(f"{clase}: {probabilidad * 100:.2f}%")

    return prediccion

    # Extraer las caracteristicas de la imagen nueva
    caracteristicas = extraer_caracteristicas_combinadas(
        ruta_imagen
    )

    # Comprobar que las dimensiones coincidan
    if caracteristicas.shape[1] != datos["num_caracteristicas"]:
        raise ValueError(
            "Las caracteristicas de la imagen no coinciden con el modelo."
        )

    # Realizar la prediccion
    prediccion = clasificador.predict(caracteristicas)[0]

    # Obtener probabilidades por categoria
    probabilidades = clasificador.predict_proba(caracteristicas)[0]

    print("\n===== RESULTADO DE KICKSMIND =====")
    print("Sneaker identificado:", prediccion)

    print("\nProbabilidades por categoria:")

    for clase, probabilidad in zip(datos["clases"], probabilidades):
        print(f"{clase}: {probabilidad * 100:.2f}%")

    return prediccion


# ==========================================
# 5. EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":

    print("\nKICKSMIND ESTA LISTO PARA CLASIFICAR.")

    imagen = r"C:\Users\luise\OneDrive\Escritorio\af1 3.jpg"

    predecir_sneaker(imagen)
