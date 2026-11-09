import cv2
import matplotlib.pyplot as plt
from preprocesamiento import preprocesar, cargar_imagen, RAIZ, EXT

resultados = []
for carpeta in sorted(p for p in RAIZ.iterdir() if p.is_dir()):
    for f in carpeta.iterdir():
        if f.suffix.lower() not in EXT:
            continue
        _, binaria, bordes = preprocesar(f)
        densidad = (bordes > 0).mean()
        m = 12
        marco = [binaria[:m, :], binaria[-m:, :], binaria[:, :m], binaria[:, -m:]]
        toca = sum((x > 0).mean() for x in marco) / 4
        resultados.append((densidad, toca, f))

resultados.sort(key=lambda r: r[0] + r[1], reverse=True)
top = resultados[:24]

print(f"{'densidad':>9} {'margen':>7}  archivo")
for d, t, f in top:
    print(f"{d:9.3f} {t:7.2f}  {f.parent.name}/{f.name}")

fig, ejes = plt.subplots(4, 6, figsize=(14, 10))
for ax, (d, t, f) in zip(ejes.ravel(), top):
    img = cv2.cvtColor(cargar_imagen(f), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{f.parent.name}\n{f.name}\nd={d:.3f} m={t:.2f}", fontsize=7)
plt.tight_layout()
plt.savefig("revision_sospechosas.png", dpi=110)
plt.show()