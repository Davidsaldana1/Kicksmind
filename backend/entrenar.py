
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features_v2 import FEATURE_COUNT_V2, extract_v2
from features_v3 import FEATURE_COUNT_V3, extract_v3

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.jfif', '.avif'}
CLASSES = ('Adidas_Campus', 'Adidas_Samba', 'Air_force_one', 'Jordan_1', 'Jordan_4')
MODEL_FILE = ROOT / 'modelo_kicksmind_v3.joblib'
REPORT_FILE = ROOT / 'resultados_validacion_v3.json'
ALL_FILE = ROOT / 'predicciones_validacion_v3.csv'
ERROR_FILE = ROOT / 'errores_clasificacion_v3.csv'
SKIPPED_FILE = ROOT / 'imagenes_omitidas_v3.csv'
MIXED_GROUPS_FILE = ROOT / 'grupos_etiquetas_distintas_v3.csv'
CACHE_DIR = ROOT / '.cache_caracteristicas_v3'


def save_csv(path: Path, rows, fields) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def perceptual_hash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))[:8, :8]
    bits = (dct > dct[1:, :].mean()).ravel().astype(np.uint8)
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')



def source_signature() -> str:
    """Si cambia el preprocesamiento, invalida el cache automaticamente."""
    fingerprint = hashlib.sha256()
    for filename in ('features.py', 'features_v2.py', 'features_v3.py'):
        fingerprint.update((ROOT / filename).read_bytes())
    return fingerprint.hexdigest()[:16]


def extract_cached(path: Path, image_sha: str, signature: str, use_cache: bool):
    """Extrae características de cada imagen una sola vez por versión del código."""
    cache_path = CACHE_DIR / f'{image_sha}_{signature}.npz'
    if use_cache and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                v2 = saved['v2'].astype(np.float32)
                v3 = saved['v3'].astype(np.float32)
                hashed = int(saved['phash'].item())
                mode = str(saved['mode'].item())
            if (v2.shape == (FEATURE_COUNT_V2,)
                    and v3.shape == (FEATURE_COUNT_V3,)
                    and np.isfinite(v2).all() and np.isfinite(v3).all()):
                return v2, v3, hashed, mode, True
        except (OSError, ValueError, KeyError):
            # Si una entrada está dañada, se vuelve a calcular.
            pass

    original2, features2 = extract_v2(path)
    original3, features3 = extract_v3(path)
    v2 = features2.ravel().astype(np.float32)
    v3 = features3.ravel().astype(np.float32)
    hashed = perceptual_hash(original2.gray)
    mode = original3.processing_mode
    if use_cache:
        temporary = CACHE_DIR / (cache_path.stem + '.tmp.npz')
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            np.savez_compressed(temporary, v2=v2, v3=v3,
                                phash=np.uint64(hashed), mode=np.asarray(mode))
            os.replace(temporary, cache_path)
        except OSError:
            # Si OneDrive bloquea el cache temporal, el entrenamiento continua.
            temporary.unlink(missing_ok=True)
    return v2, v3, hashed, mode, False

def collect(dataset: Path, use_cache: bool = True):
    if not dataset.is_dir():
        raise FileNotFoundError(f'No existe el dataset: {dataset}. Copia tu DATA-SET actualizado junto a entrenar.py o usa --dataset.')
    image_paths = sorted(p for sub in dataset.iterdir() if sub.is_dir() and sub.name in CLASSES
                         for p in sub.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not image_paths:
        raise ValueError('No se encontraron imágenes en las cinco carpetas esperadas.')

    seen = {}
    valid = []
    skipped = []
    image_hashes = {}
    for path in image_paths:
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        image_hashes[path] = key
        if key in seen:
            reason = 'Duplicado exacto de ' + str(seen[key])
            if path.parent != seen[key].parent:
                reason = 'ALERTA: copia exacta en otra clase: ' + str(seen[key])
            skipped.append({'ruta_imagen': str(path.resolve()), 'motivo': reason})
        else:
            valid.append(path)
            seen[key] = path

    Xv2, Xv3, labels, hashes, paths, preprocess_info = [], [], [], [], [], []
    signature = source_signature()
    reused = computed = 0
    for n, path in enumerate(valid, 1):
        try:
            v2, v3, phash, mode, is_cached = extract_cached(
                path, image_hashes[path], signature, use_cache)
        except Exception as error:
            skipped.append({'ruta_imagen': str(path.resolve()), 'motivo': str(error)})
            print(f'[AVISO] Omitida {path.name}: {error}', flush=True)
            continue
        Xv2.append(v2)
        Xv3.append(v3)
        labels.append(path.parent.name)
        hashes.append(phash)
        paths.append(path)
        preprocess_info.append(mode)
        reused += int(is_cached)
        computed += int(not is_cached)
        if n % 50 == 0 or n == len(valid):
            print(f'Imágenes procesadas: {n}/{len(valid)} | cache: {reused} | nuevas: {computed}', flush=True)
    print(f'Cache reutilizado: {reused}; características calculadas: {computed}', flush=True)

    save_csv(SKIPPED_FILE, skipped, ['ruta_imagen', 'motivo'])
    Xv2, Xv3, y = np.asarray(Xv2, dtype=np.float32), np.asarray(Xv3, dtype=np.float32), np.asarray(labels)
    if len(y) < 20 or Xv2.shape[1] != FEATURE_COUNT_V2 or Xv3.shape[1] != FEATURE_COUNT_V3:
        raise ValueError('No hay imágenes válidas suficientes o las características no tienen el tamaño esperado.')
    if set(y) != set(CLASSES) or min(Counter(y).values()) < 5:
        raise ValueError('El dataset debe contener las cinco categorías con al menos cinco imágenes válidas por clase.')

    # Agrupar copias casi idénticas en cualquier clase, para reducir fuga de datos.
    parent = list(range(len(y)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if (hashes[i] ^ hashes[j]).bit_count() <= 5:
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[ri] = rj
    groups = np.array([root(i) for i in range(len(y))])
    print('Dataset:', dataset)
    print('Imágenes válidas:', len(y), '| Grupos visuales:', len(set(groups)))
    print('Categorías:', dict(Counter(map(str, y))))
    print('Preprocesamiento V3:', dict(Counter(preprocess_info)))
    if skipped:
        print(f'[AVISO] Se omitieron {len(skipped)} archivos. Revisa {SKIPPED_FILE.name}')
    # La agrupación visual puede unir clases distintas: señalarlo al revisar etiquetas.
    by_group = {}
    for label, group in zip(y, groups):
        by_group.setdefault(int(group), set()).add(str(label))
    mixed = sum(len(labels) > 1 for labels in by_group.values())
    mixed_rows = [
        {'grupo': int(group), 'imagen': path.name, 'ruta_imagen': str(path.resolve()), 'categoria': str(label)}
        for group, label, path in zip(groups, y, paths)
        if len(by_group[int(group)]) > 1
    ]
    save_csv(MIXED_GROUPS_FILE, mixed_rows, ['grupo', 'imagen', 'ruta_imagen', 'categoria'])
    if mixed:
        print(f'[AVISO] {mixed} grupos tienen clases distintas. Revisa {MIXED_GROUPS_FILE.name}.')
    return Xv2, Xv3, y, groups, paths, preprocess_info, len(skipped), mixed


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=3000, random_state=42))


def make_svm():
    return make_pipeline(StandardScaler(), SVC(kernel='rbf', C=2.0, gamma='scale',
                                                probability=True, random_state=42))


def train_and_evaluate(Xv2, Xv3, y, groups, paths, modes, skipped_count, mixed_count):
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(Xv3, y, groups))
    candidates = [
        ('Regresión logística V2 (base)', 'v2', make_logreg, Xv2),
        ('Regresión logística V3', 'v3', make_logreg, Xv3),
        ('SVM RBF V3', 'v3', make_svm, Xv3),
    ]
    results = {}
    predictions_by_name = {}
    for name, extractor, factory, X in candidates:
        predictions = np.empty(y.shape, dtype=y.dtype)
        scores = np.empty(len(y), dtype=np.float32)
        folds = np.zeros(len(y), dtype=int)
        print(f'\nIniciando {name} (5 particiones)...', flush=True)
        for part, (train_idx, test_idx) in enumerate(splits, 1):
            print(f'  Partición {part}/5: entrenando...', flush=True)
            model = factory()
            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[test_idx])
            predictions[test_idx] = predicted
            scores[test_idx] = model.predict_proba(X[test_idx]).max(axis=1)
            folds[test_idx] = part
            print(f'  Partición {part}/5: completada.', flush=True)
        res = {
            'extractor': extractor,
            'accuracy': float(accuracy_score(y, predictions)),
            'f1_macro': float(f1_score(y, predictions, average='macro')),
            'labels': list(CLASSES),
            'confusion_matrix': confusion_matrix(y, predictions, labels=CLASSES).tolist(),
            'classification_report': classification_report(y, predictions, labels=CLASSES, output_dict=True, zero_division=0),
        }
        results[name] = res
        predictions_by_name[name] = (predictions, scores, folds)
        print(f'\n{name}: accuracy {res["accuracy"]:.2%} | F1 macro {res["f1_macro"]:.2%}', flush=True)

    # Preferir la calidad equilibrada de todas las clases (F1 macro).
    winner_name = max(results, key=lambda name: (results[name]['f1_macro'], results[name]['accuracy']))
    winner_config = next(item for item in candidates if item[0] == winner_name)
    _, extractor, factory, X = winner_config
    preds, scores, folds = predictions_by_name[winner_name]
    rows = []
    for path, real, predicted, score, fold, mode in zip(paths, y, preds, scores, folds, modes):
        rows.append({
            'imagen': path.name, 'ruta_imagen': str(path.resolve()),
            'categoria_real': str(real), 'prediccion': str(predicted),
            'correcta': 'SI' if real == predicted else 'NO', 'particion': int(fold),
            'puntuacion_modelo_pct': f'{score*100:.1f}', 'preprocesamiento': mode,
        })
    fields = ['imagen', 'ruta_imagen', 'categoria_real', 'prediccion', 'correcta',
              'particion', 'puntuacion_modelo_pct', 'preprocesamiento']
    save_csv(ALL_FILE, rows, fields)
    errors = [row for row in rows if row['correcta'] == 'NO']
    save_csv(ERROR_FILE, errors, fields)
    print(f'\nModelo elegido: {winner_name} ({extractor})')
    print(f'Errores de validación: {len(errors)} de {len(rows)}', flush=True)
    print('Principales confusiones:', Counter((r['categoria_real'], r['prediccion']) for r in errors).most_common(5))

    # Entrenamiento final sobre TODO el conjunto, nunca evaluado sobre él mismo.
    final_model = factory()
    final_model.fit(X, y)
    package = {'version': 3, 'extractor': extractor, 'algorithm': winner_name,
               'feature_count': int(X.shape[1]), 'model': final_model}
    if MODEL_FILE.is_file():
        backup_file = ROOT / 'modelo_kicksmind_v3_anterior.joblib'
        shutil.copy2(MODEL_FILE, backup_file)
        print(f'[RESPALDO] Modelo anterior: {backup_file.name}', flush=True)
    joblib.dump(package, MODEL_FILE, compress=3)
    report = {
        'metodo': 'Validación cruzada estratificada por grupos, 5 particiones compartidas',
        'advertencia': 'Evaluación exploratoria; modelo elegido sobre las mismas particiones. Necesita prueba externa independiente.',
        'imagenes_validas': len(y), 'grupos_similares': len(set(groups)),
        'grupos_con_etiquetas_diferentes': mixed_count, 'archivos_omitidos': skipped_count,
        'categorias': dict(Counter(map(str, y))), 'preprocesamiento_v3': dict(Counter(modes)),
        'modelo_elegido': winner_name, 'extractor_elegido': extractor,
        'resultados': results,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] Modelo guardado: {MODEL_FILE.name}')
    print(f'[OK] Evaluación guardada: {REPORT_FILE.name}')
    print(f'[OK] Errores: {ERROR_FILE.name}')
    return report


def main():
    parser = argparse.ArgumentParser(description='Entrenar KicksMind V3 y comparar con V2')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'DATA-SET')
    parser.add_argument('--sin-cache', action='store_true', help='Recalcular todas las características.')
    args = parser.parse_args()
    Xv2, Xv3, y, groups, paths, modes, skipped, mixed = collect(args.dataset, use_cache=not args.sin_cache)
    train_and_evaluate(Xv2, Xv3, y, groups, paths, modes, skipped, mixed)


if __name__ == '__main__':
    main()
