
import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. CARGAR CARACTERISTICAS HOG
# ==========================================

X = np.load("X_hog.npy")
y = np.load("y.npy")

print("HOG cargado correctamente")
print("Dimensiones originales:", X.shape)


# ==========================================
# 2. CONFIGURAR VALIDACION CRUZADA
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 3. CONFIGURAR COMPONENTES PCA
# ==========================================

componentes = [25, 50, 100, 150]


# ==========================================
# 4. CONFIGURAR CLASIFICADORES
# ==========================================

modelos = {
    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ==========================================
# 5. REALIZAR EXPERIMENTOS
# ==========================================

print("\n===== EXPERIMENTOS CON PCA =====")

resultados = []

for cantidad in componentes:

    print(f"\n\n===== PCA: {cantidad} COMPONENTES =====")

    for nombre, clasificador in modelos.items():

        # PCA se entrena dentro de cada particion
        modelo = make_pipeline(
            StandardScaler(),
            PCA(n_components=cantidad, random_state=42),
            clasificador
        )

        pred = cross_val_predict(
            modelo,
            X,
            y,
            cv=cv
        )

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

        resultados.append((cantidad, nombre, accuracy))

        print(f"\nModelo: {nombre}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")


# ==========================================
# 6. IDENTIFICAR MEJOR CONFIGURACION
# ==========================================

mejor = max(resultados, key=lambda x: x[2])

print("\n================================")
print("       MEJOR RESULTADO PCA")
print("================================")

print("Componentes:", mejor[0])
print("Clasificador:", mejor[1])
print(f"Accuracy: {mejor[2]:.3f}")

print("\nEXPERIMENTO FINALIZADO")
