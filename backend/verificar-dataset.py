from PIL import Image
from preprocesamiento import RAIZ, EXT

conteos = {}
for carpeta in sorted(p for p in RAIZ.iterdir() if p.is_dir()):
    archivos = [f for f in carpeta.iterdir() if f.suffix.lower() in EXT]
    danadas = 0
    for f in archivos:
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception:
            danadas += 1
            print("Dañada:", f)
    conteos[carpeta.name] = len(archivos) - danadas
    print(f"{carpeta.name:25s} {len(archivos):4d} imágenes ({danadas} dañadas)")

menor, mayor = min(conteos.values()), max(conteos.values())
print(f"\nTotal: {sum(conteos.values())} imágenes en {len(conteos)} clases")
print(f"Clase más chica: {menor} | clase más grande: {mayor}")
if menor < 30:
    print("Aviso: hay clases con menos de 30 imágenes, conviene reponerlas.")
if mayor > 1.5 * menor:
    print("Aviso: las clases están desbalanceadas (la más grande supera por 50% a la más chica).")