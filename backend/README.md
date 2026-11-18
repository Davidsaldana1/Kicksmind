# KicksMind V3 — mejora del reconocimiento

Proyecto Python editable en Visual Studio Code. **No reemplaza tu V2 original** ni incluye un `.exe` todavía.

## 1. Qué cambió

- **Preprocesamiento V3** (`features_v3.py`): intenta aislar el sneaker **solo si detecta un fondo uniforme**. Si no, utiliza la foto completa (sin asumir que la segmentación es correcta).
- **Tamaño 256 × 256 sin deformar**: ajusta proporcionalmente la imagen y añade bordes neutros cuando hace falta.
- **HOG multiescala, LBP, Box Counting, HSV**: se calculan siempre desde el mismo preprocesamiento tanto al entrenar como al reconocer.
- **Comparación justa de métodos**: compara Regresión logística V2, Regresión logística V3 y SVM V3 usando las mismas cinco particiones y agrupando fotografías muy similares.
- **Selección automática**: guarda en `modelo_kicksmind_v3.joblib` el método con mayor F1-score macro en la validación. El archivo indica qué extractor requiere, evitando cargar accidentalmente el extractor V2 para un modelo V3.
- **Diagnóstico**: genera resultados JSON y archivos CSV de predicciones, errores, archivos omitidos y grupos de etiquetas potencialmente conflictivas.
- **Interfaz**: conserva el inicio de sesión, muestra el algoritmo utilizado y si se aisló el fondo.

## 2. Ejecutar de inmediato en VS Code

Extrae el ZIP y abre la carpeta **KicksMind_V3** en VS Code.

```powershell
py -m pip install -r requirements.txt
py app.py
```

Credenciales actuales de demostración: `admin` / `0101`.

> El ZIP trae un modelo V3 **provisional, entrenado con las 304 imágenes archivadas de V2**, para que puedas abrirlo y probarlo. **NO está entrenado con tus 496 imágenes actuales**. La contraseña corta no sirve para proteger datos sensibles.

## 3. Usar TUS 496 imágenes y entrenar de nuevo (muy importante)

1. **No borres ni modifiques tu carpeta V2 original.**
2. De tu carpeta `KicksMind_V2`, copia la carpeta completa `DATA-SET` (la que ahora contiene 496 fotos) y pégala dentro de `KicksMind_V3`.
3. Comprueba que quede así:

```text
KicksMind_V3/
  app.py
  entrenar.py
  features.py
  features_v2.py
  features_v3.py
  modelo_kicksmind_v3.joblib       <- provisional, será actualizado
  DATA-SET/
    Adidas_Campus/
    Adidas_Samba/
    Air_force_one/
    Jordan_1/
    Jordan_4/
```

4. Cierra la aplicación si está abierta y ejecuta:

```powershell
py entrenar.py
```

5. Cuando aparezca `[OK] Modelo guardado`, ejecuta:

```powershell
py app.py
```

El entrenamiento puede demorar, especialmente al evaluar tres modelos. No se interrumpe por una imagen ilegible: la registra en `imagenes_omitidas_v3.csv` y continúa.

También puedes señalar el dataset en su ubicación original, **sin copiarlo**:

```powershell
py entrenar.py --dataset "C:\ruta\a\KicksMind_V2\DATA-SET"
```

## 4. Resultados de la prueba disponible aquí

En el conjunto archivado de **304 imágenes** (NO el dataset más reciente de 496), bajo cinco particiones iguales para los tres métodos:

| Método | Accuracy | F1-score macro |
|---|---:|---:|
| Regresión logística V2 | 61.51% | 61.30% |
| Regresión logística V3 | **76.32%** | **75.38%** |
| SVM RBF V3 | 72.37% | 72.33% |

Estos resultados son **exploratorios**: la evaluación se utilizó también para elegir el método ganador, y todavía no existe prueba con fotografías nuevas totalmente independientes. El rendimiento con el dataset actualizado puede cambiar. El archivo `resultados_validacion_v3.json` tiene detalles y matrices de confusión.

**Aviso:** en las 304 imágenes archivadas surgieron tres grupos visualmente similares con etiquetas diferentes. El entrenamiento actualizado genera `grupos_etiquetas_distintas_v3.csv` para inspeccionarlos. Que dos imágenes estén agrupadas no prueba que estén mal etiquetadas.

## 5. Qué archivos genera `entrenar.py`

- `modelo_kicksmind_v3.joblib`: modelo que utilizará `app.py`.
- `resultados_validacion_v3.json`: comparativa y matrices de confusión.
- `predicciones_validacion_v3.csv`: predicciones hechas en los folds de validación.
- `errores_clasificacion_v3.csv`: imágenes clasificadas incorrectamente.
- `imagenes_omitidas_v3.csv`: archivos que no pudieron procesarse o eran duplicados exactos.
- `grupos_etiquetas_distintas_v3.csv`: pares/grupos de imágenes parecidas con categorías diferentes.

Los CSV se guardan en UTF-8 con separador `;`, compatibles con Excel en español.

## 6. Probar una fotografía sin abrir interfaz

```powershell
py prueba.py "C:\ruta\a\mi_tenis.jpg"
```

> Un porcentaje cercano al 100% puede ser incorrecto. La puntuación del modelo **no es una garantía de acierto**.

## 7. Crear el `.exe` cuando termines

```powershell
py -m pip install pyinstaller
```

Después ejecuta `build_exe.bat`. Comparte **toda** la carpeta `dist\KicksMind`, no solo `KicksMind.exe`. **No construyas el EXE** antes de entrenar con tus 496 imágenes y probar varias imágenes nuevas.

## 8. Limitaciones y honestidad de los resultados

- Con fondos complejos (escalones, personas, calles), el preprocesamiento **no garantiza aislar el sneaker**: conserva la foto completa.
- Los cinco modelos reconocibles son Adidas Campus, Adidas Samba, Air Force 1, Jordan 1 y Jordan 4. Un sneaker desconocido será asignado a uno de ellos; no hay detector de "modelo desconocido".
- El preprocesamiento V3 cambia las características; por eso **no debes usar el viejo modelo V2 directamente como V3**. La aplicación lee la versión de extractor guardada con el nuevo modelo.
- Las etiquetas y la calidad de las imágenes siguen siendo críticas. Un siguiente paso recomendable es separar 20–30 fotos nuevas por clase como un **test independiente**, sin usarlas para entrenar ni elegir parámetros.
- No distribuyas el dataset públicamente sin comprobar los derechos de uso de las imágenes.
- Abre modelos `.joblib` solo de fuentes confiables.
