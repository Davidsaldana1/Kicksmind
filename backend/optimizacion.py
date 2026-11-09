
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cargar características HOG y etiquetas
X = np.load("X_combinado.npy")
y = np.load("y.npy")

# Validación cruzada de 5 partes
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados = []

print("===== OPTIMIZACIÓN DE MODELOS KICKSMIND =====")

# --------------------------------
# 1. PROBAR DIFERENTES VALORES DE KNN
# --------------------------------

valores_k = [3, 5, 7, 9, 11, 13, 15]

for k in valores_k:

    modelo = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    predicciones = cross_val_predict(modelo, X, y, cv=cv)

    accuracy = accuracy_score(y, predicciones)
    precision = precision_score(y, predicciones, average="macro", zero_division=0)
    recall = recall_score(y, predicciones, average="macro", zero_division=0)
    f1 = f1_score(y, predicciones, average="macro", zero_division=0)

    resultados.append(("KNN", k, accuracy, precision, recall, f1))

    print(f"\nKNN - K={k}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


# --------------------------------
# 2. PROBAR DIFERENTES RANDOM FOREST
# --------------------------------

valores_arboles = [100, 200, 300, 500]

for cantidad in valores_arboles:

    modelo = RandomForestClassifier(
        n_estimators=cantidad,
        random_state=42
    )

    predicciones = cross_val_predict(modelo, X, y, cv=cv)

    accuracy = accuracy_score(y, predicciones)
    precision = precision_score(y, predicciones, average="macro", zero_division=0)
    recall = recall_score(y, predicciones, average="macro", zero_division=0)
    f1 = f1_score(y, predicciones, average="macro", zero_division=0)

    resultados.append(("Random Forest", cantidad, accuracy, precision, recall, f1))

    print(f"\nRandom Forest - {cantidad} árboles")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


# --------------------------------
# 3. MOSTRAR EL MEJOR RESULTADO
# --------------------------------

mejor = max(resultados, key=lambda x: x[2])

print("\n================================")
print("       MEJOR CONFIGURACIÓN")
print("================================")

print("Modelo:", mejor[0])
print("Parámetro:", mejor[1])
print(f"Accuracy: {mejor[2]:.3f}")
print(f"Precision: {mejor[3]:.3f}")
print(f"Recall: {mejor[4]:.3f}")
print(f"F1-score: {mejor[5]:.3f}")

print("\nOPTIMIZACIÓN FINALIZADA")
