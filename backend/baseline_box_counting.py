import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocesamiento import preprocesar, RAIZ, EXT
from box_counting import box_counting, recortar_y_normalizar


def caracteristicas_bc(ruta):
    _, binaria, bordes = preprocesar(ruta)
    d_b, N_b = box_counting(recortar_y_normalizar(bordes))
    d_s, _ = box_counting(recortar_y_normalizar(binaria))
    return [d_b, d_s] + list(np.log(np.maximum(N_b, 1)))


X, y = [], []
for carpeta in sorted(p for p in RAIZ.iterdir() if p.is_dir()):
    for f in sorted(carpeta.iterdir()):
        if f.suffix.lower() in EXT:
            X.append(caracteristicas_bc(f))
            y.append(carpeta.name)
X, y = np.array(X), np.array(y)
np.save("X_bc.npy", X)
np.save("y.npy", y)
print("Matriz de características:", X.shape)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelos = {
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
nombres = sorted(set(y))
for nombre, modelo in modelos.items():
    pred = cross_val_predict(modelo, X, y, cv=cv)
    print(f"\n{nombre}: accuracy = {accuracy_score(y, pred):.3f}")
    print(nombres)
    print(confusion_matrix(y, pred, labels=nombres))