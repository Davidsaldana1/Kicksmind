
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocesamiento import preprocesar, RAIZ, EXT
from box_counting import box_counting, recortar_y_normalizar
from hog_lbp import extraer_hog, extraer_lbp


# ==========================================
# 1. EXTRAER CARACTERISTICAS BOX COUNTING
# ==========================================

def caracteristicas_bc(ruta):

    _, binaria, bordes = preprocesar(ruta)

    d_b, N_b = box_counting(
        recortar_y_normalizar(bordes)
    )

    d_s, _ = box_counting(
        recortar_y_normalizar(binaria)
    )

    return [d_b, d_s] + list(
        np.log(np.maximum(N_b, 1))
    )


# ==========================================
# 2. RECORRER EL DATASET Y EXTRAER DATOS
# ==========================================

X_bc = []
X_hog = []
X_lbp = []
y = []

print("EXTRAYENDO CARACTERISTICAS...\n")

for carpeta in sorted(p for p in RAIZ.iterdir() if p.is_dir()):

    imagenes = sorted(
        f for f in carpeta.iterdir()
        if f.suffix.lower() in EXT
    )

    for ruta in imagenes:

        # Extraemos las caracteristicas de cada metodo
        bc = caracteristicas_bc(ruta)
        hog = extraer_hog(ruta)
        lbp = extraer_lbp(ruta)

        X_bc.append(bc)
        X_hog.append(hog)
        X_lbp.append(lbp)

        # Guardamos la marca y modelo como etiqueta
        y.append(carpeta.name)

    print(f"{carpeta.name}: {len(imagenes)} imagenes")


# Convertimos las listas en matrices
X_bc = np.array(X_bc)
X_hog = np.array(X_hog)
X_lbp = np.array(X_lbp)
y = np.array(y)


# ==========================================
# 3. COMBINAR LOS TRES METODOS
# ==========================================

X_combinado = np.concatenate(
    [X_bc, X_hog, X_lbp],
    axis=1
)


# Guardamos las matrices para reutilizarlas
np.save("X_bc.npy", X_bc)
np.save("X_hog.npy", X_hog)
np.save("X_lbp.npy", X_lbp)
np.save("X_combinado.npy", X_combinado)
np.save("y.npy", y)


print("\n===== DIMENSIONES DE LAS MATRICES =====")

print("Box Counting:", X_bc.shape)
print("HOG:", X_hog.shape)
print("LBP:", X_lbp.shape)
print("COMBINADO:", X_combinado.shape)


# ==========================================
# 4. PREPARAR LOS EXPERIMENTOS
# ==========================================

experimentos = {
    "BOX COUNTING": X_bc,
    "HOG": X_hog,
    "LBP": X_lbp,
    "COMBINADO (BC + HOG + LBP)": X_combinado
}


# ==========================================
# 5. CONFIGURAR LOS CLASIFICADORES
# ==========================================

modelos = {
    "KNN (k=5)": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "RANDOM FOREST": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ==========================================
# 6. VALIDACION CRUZADA
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

nombres = sorted(set(y))

print("\n======================================")
print("       RESULTADOS DE KICKSMIND")
print("======================================")


for nombre_exp, X in experimentos.items():

    print(f"\n\n===== {nombre_exp} =====")

    for nombre_modelo, modelo in modelos.items():

        # Predicciones mediante validacion cruzada
        pred = cross_val_predict(
            modelo,
            X,
            y,
            cv=cv
        )

        # Calcular metricas
        accuracy = accuracy_score(y, pred)

        precision = precision_score(
            y, pred, average="macro", zero_division=0
        )

        recall = recall_score(
            y, pred, average="macro", zero_division=0
        )

        f1 = f1_score(
            y, pred, average="macro", zero_division=0
        )

        matriz = confusion_matrix(
            y, pred, labels=nombres
        )

        print(f"\nModelo: {nombre_modelo}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

        print("\nMatriz de confusion:")
        print("Orden de clases:", nombres)
        print(matriz)


print("\n======================================")
print("       EXPERIMENTOS FINALIZADOS")
print("======================================")
