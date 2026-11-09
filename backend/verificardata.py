
from preprocesamiento import RAIZ, EXT, cargar_imagen

print("===== VERIFICACIÓN DATASET KICKSMIND =====")

print("\nRuta que está leyendo Python:")
print(RAIZ)

print("\nCarpetas encontradas:")

for carpeta in RAIZ.iterdir():
    print(carpeta.name)

total = 0
errores = 0

for carpeta in sorted(p for p in RAIZ.iterdir() if p.is_dir()):

    imagenes = [
        archivo for archivo in carpeta.iterdir()
        if archivo.suffix.lower() in EXT
    ]

    cantidad = len(imagenes)
    total += cantidad

    imagenes_danadas = []

    for imagen in imagenes:
        try:
            img = cargar_imagen(imagen)

            if img is None:
                imagenes_danadas.append(imagen.name)

        except Exception:
            imagenes_danadas.append(imagen.name)

    errores += len(imagenes_danadas)

    print(f"\nCategoría: {carpeta.name}")
    print(f"Imágenes encontradas: {cantidad}")
    print(f"Imágenes con problemas: {len(imagenes_danadas)}")

    for nombre in imagenes_danadas:
        print(" -", nombre)

print("\n================================")
print("TOTAL DE IMÁGENES:", total)
print("TOTAL DE IMÁGENES CON PROBLEMAS:", errores)
print("VERIFICACIÓN FINALIZADA")
