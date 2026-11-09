
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. CARGAR LAS CARACTERISTICAS
# ==========================================

X_bc = np.load("X_bc.npy")
X_hog = np.load("X_hog.npy")
X_lbp = np.load("X_lbp.npy")
y = np.load("y.npy")

print("Caracteristicas cargadas correctamente")
print("Box Counting:", X_bc.shape)
print("HOG:", X_hog.shape)
print("LBP:", X_lbp.shape)


# ==========================================
# 2. NORMALIZAR CADA GRUPO
# ==========================================

# Cada imagen tendrá un vector de longitud 1
bc_norm = normalize(X_bc, norm="l2")
hog_norm = normalize(X_hog, norm="l2")
lbp_norm = normalize(X_lbp, norm="l2")


# ==========================================
# 3. CONFIGURAR DIFERENTES PESOS
# ==========================================

configuraciones = {
    "Equilibrado": (1.0, 1.0, 1.0),
    "HOG reducido": (1.0, 0.5, 1.0),
    "LBP reforzado": (1.0, 1.0, 2.0),
    "Box Counting reforzado": (2.0, 1.0, 1.0),
    "HOG muy reducido": (1.0, 0.25, 1.0)
}


# ==========================================
# 4. PREPARAR VALIDACION CRUZADA
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

modelo = KNeighborsClassifier(n_neighbors=5)


# ==========================================
# 5. PROBAR CADA CONFIGURACION
# ==========================================

print("\n================================")
print(" EXPERIMENTO DE PONDERACION")
print("================================")

resultados = []

for nombre, (peso_bc, peso_hog, peso_lbp) in configuraciones.items():

    # Unimos las caracteristicas aplicando sus pesos
    X = np.concatenate(
        [
            bc_norm * peso_bc,
            hog_norm * peso_hog,
            lbp_norm * peso_lbp
        ],
        axis=1
    )

    # Entrenamiento y predicciones
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

    resultados.append((nombre, accuracy))

    print(f"\nConfiguracion: {nombre}")
    print(f"Pesos: BC={peso_bc}, HOG={peso_hog}, LBP={peso_lbp}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


# ==========================================
# 6. IDENTIFICAR EL MEJOR RESULTADO
# ==========================================

mejor = max(resultados, key=lambda x: x[1])

print("\n================================")
print(" MEJOR CONFIGURACION")
print("================================")

print("Configuracion:", mejor[0])
print(f"Accuracy: {mejor[1]:.3f}")

print("\nEXPERIMENTO FINALIZADO")
