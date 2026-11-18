"""Prueba una imagen sin GUI: py prueba.py "C:\\fotos\\foto.jpg" """
import sys
import joblib
from app import find_model
from features_v2 import extract_v2
from features_v3 import extract_v3


def main():
    model_file = find_model()
    if not model_file.is_file():
        raise FileNotFoundError('No existe el modelo. Ejecuta primero: py entrenar.py')
    bundle = joblib.load(model_file)
    if isinstance(bundle, dict) and bundle.get('version') == 3:
        model = bundle['model']
        extractor = bundle['extractor']
        print('Modelo:', bundle['algorithm'], '| extractor:', extractor)
    else:
        model, extractor = bundle, 'v2'
        print('Modelo V2 de respaldo')
    print('Características:', model.n_features_in_)
    if len(sys.argv) < 2:
        print('Uso: py prueba.py "ruta-de-la-foto.jpg"')
        return
    image, vector = extract_v3(sys.argv[1]) if extractor == 'v3' else extract_v2(sys.argv[1])
    print('Predicción:', model.predict(vector)[0])
    if extractor == 'v3':
        print('Preprocesamiento:', image.processing_mode)
    for cls, p in zip(model.classes_, model.predict_proba(vector)[0]):
        print(f'  {cls}: {p * 100:.1f}%')


if __name__ == '__main__':
    main()
