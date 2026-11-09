
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

# 1. Cargar las características combinadas y las etiquetas
X = np.load("X_combinado.npy")
y = np.load("y.npy")

# 2. Crear el modelo Random Forest
modelo = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# 3. Entrenar el modelo
modelo.fit(X, y)

# 4. Guardar el modelo entrenado
joblib.dump(modelo, "modelo_final_kicksmind.joblib")

print("===== MODELO FINAL KICKSMIND =====")
print("Algoritmo: Random Forest")
print("Número de árboles: 300")
print("Características:", X.shape[1])
print("Imágenes utilizadas:", X.shape[0])
print("Modelo guardado correctamente.")
