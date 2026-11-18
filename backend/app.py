"""KicksMind V3 · reconocimiento de sneakers.

La interfaz detecta qué extractor (V2/V3) se utilizó al entrenar.
La autenticación es local y solo para demostración académica.
"""
import hashlib
import hmac
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
from PIL import Image, ImageTk

from features import BOX_SIZES
from features_v2 import FEATURE_COUNT_V2, extract_v2
from features_v3 import FEATURE_COUNT_V3, extract_v3

# Estilo streetwear del proyecto original.
BG = "#090E15"
PANEL = "#141E29"
INNER = "#0D1520"
BORDER = "#29394A"
GREEN = "#ABFF35"
RED = "#FF526A"
WHITE = "#F3F7FA"
MUTED = "#9AAABC"
YELLOW = "#FFCC6D"
FONT = "Segoe UI"
MODEL_NAME = "modelo_kicksmind_v3.joblib"
FALLBACK_MODEL_NAME = "modelo_kicksmind_v2.joblib"

# Acceso LOCAL de demostración: usuario admin, contraseña documentada en README.
# PBKDF2 evita guardar la clave como texto plano en el código fuente;
# no convierte una app de escritorio en un sistema de autenticación seguro.
PASSWORD_SALT = b"kicks"
PASSWORD_HASH = "027a280e9abb9ef6b35980089fa10bd737428fd4fd81843ed34a96e7fd85e005"


def find_model() -> Path:
    """Ubica primero V3 y, si no existe, permite usar el modelo V2 de respaldo."""
    here = Path(__file__).resolve().parent
    directories = [here]
    if getattr(sys, "frozen", False):
        directories = [Path(sys.executable).resolve().parent,
                       Path(getattr(sys, "_MEIPASS", here)), here]
    for filename in (MODEL_NAME, FALLBACK_MODEL_NAME):
        for directory in directories:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return here / MODEL_NAME


def validate_login(username: str, password: str) -> bool:
    if username.strip() != "admin":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), PASSWORD_SALT, 200_000
    )
    return hmac.compare_digest(digest.hex(), PASSWORD_HASH)


def label(parent, text, size=10, color=WHITE, bold=False, bg=PANEL, **kwargs):
    return tk.Label(
        parent, text=text, fg=color, bg=bg,
        font=(FONT, size, "bold" if bold else "normal"), **kwargs,
    )


def button(parent, text, action, background=GREEN, foreground=BG, **kwargs):
    return tk.Button(
        parent, text=text, command=action, bg=background, fg=foreground,
        activebackground=background, activeforeground=foreground,
        font=(FONT, 10, "bold"), padx=12, pady=10,
        relief="flat", bd=0, cursor="hand2", **kwargs,
    )


def panel(parent, title: str):
    box = tk.Frame(
        parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1,
    )
    label(box, title, size=11, bold=True).pack(anchor="w", padx=14, pady=(14, 12))
    return box


class KicksMind:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KicksMind | Sneaker Recognition")
        self.root.configure(bg=BG)
        self.root.geometry("460x490")
        self.root.minsize(420, 450)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.model = None
        self.extractor = None
        self.algorithm = None
        self.image_path: Path | None = None
        self.photos = {}  # Mantiene referencias ImageTk para evitar que Tk las borre.
        self.counts = None
        self.show_login()

    def show_login(self):
        self.root.minsize(420, 450)
        self.root.geometry("460x490")
        self.root.unbind("<Return>")
        for child in self.root.winfo_children():
            child.destroy()
        self.image_path = None
        self.photos.clear()
        self.counts = None

        frame = tk.Frame(self.root, bg=BG)
        frame.pack(expand=True, fill="both", padx=32, pady=25)
        label(frame, "KICKSMIND", 31, GREEN, True, BG).pack(pady=(23, 1))
        label(frame, "SNEAKER RECOGNITION SYSTEM", 10, MUTED, True, BG).pack()

        form = tk.Frame(frame, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        form.pack(fill="x", pady=(30, 0), ipadx=22, ipady=18)
        label(form, "ACCESO DE ADMINISTRADOR", 12, WHITE, True).pack(pady=(5, 17))
        label(form, "Usuario", 10, MUTED).pack(anchor="w")
        self.username = tk.Entry(
            form, font=(FONT, 11), bg=INNER, fg=WHITE, insertbackground=WHITE,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
        )
        self.username.pack(fill="x", ipady=8, pady=(5, 13))
        label(form, "Contraseña", 10, MUTED).pack(anchor="w")
        self.password = tk.Entry(
            form, show="•", font=(FONT, 11), bg=INNER, fg=WHITE,
            insertbackground=WHITE, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.password.pack(fill="x", ipady=8, pady=(5, 20))
        button(form, "INICIAR SESIÓN", self.do_login).pack(fill="x")
        label(frame, "Acceso local para demostración académica", 9, MUTED, bg=BG).pack(pady=19)
        self.root.bind("<Return>", lambda _event: self.do_login())
        self.username.focus_set()

    def do_login(self):
        if not validate_login(self.username.get(), self.password.get()):
            self.password.delete(0, tk.END)
            messagebox.showerror("Acceso denegado", "Usuario o contraseña incorrectos.")
            return
        self.root.unbind("<Return>")
        for child in self.root.winfo_children():
            child.destroy()
        self.root.geometry("1180x780")
        self.root.minsize(1000, 700)
        self.build_workspace()

    def build_workspace(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=22, pady=(17, 5))
        label(top, "KICKSMIND", 26, GREEN, True, BG).pack(side="left")
        label(top, "SNEAKER RECOGNITION", 9, MUTED, True, BG).pack(side="left", padx=14)
        button(top, "Cerrar sesión", self.show_login, PANEL, WHITE).pack(side="right")

        label(self.root, "Panel de análisis", 18, WHITE, True, BG).pack(
            anchor="w", padx=23, pady=(4, 0)
        )
        label(
            self.root,
            "Imagen → ajuste proporcional + segmentación prudente → BC / HOG / LBP / HSV → clasificación",
            9, MUTED, bg=BG,
        ).pack(anchor="w", padx=23, pady=(4, 14))

        workspace = tk.Frame(self.root, bg=BG)
        workspace.pack(fill="both", expand=True, padx=17, pady=(0, 8))
        for i in range(3):
            workspace.columnconfigure(i, weight=1, uniform="sections")
        workspace.rowconfigure(0, weight=1)

        left = panel(workspace, "01   Selecciona tu sneaker")
        center = panel(workspace, "02   Procesamiento y características")
        right = panel(workspace, "03   Resultado de la clasificación")
        for index, p in enumerate((left, center, right)):
            p.grid(row=0, column=index, sticky="nsew", padx=5)

        self.build_image_panel(left)
        self.build_features_panel(center)
        self.build_result_panel(right)

        self.status = label(
            self.root, "Sistema listo · selecciona una imagen para comenzar",
            9, MUTED, True, BG,
        )
        self.status.pack(pady=(1, 11))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Kicks.Horizontal.TProgressbar", troughcolor=INNER,
            background=GREEN, bordercolor=BORDER,
            lightcolor=GREEN, darkcolor=GREEN,
        )
        self.root.after(80, self.draw_plot)

    def build_image_panel(self, parent):
        preview = tk.Frame(
            parent, bg=INNER, highlightbackground=BORDER, highlightthickness=1,
        )
        preview.pack(fill="both", expand=True, padx=13, pady=(0, 12))
        self.preview = label(
            preview, "Selecciona una fotografía\npara comenzar", 11, MUTED, bg=INNER
        )
        self.preview.pack(expand=True, fill="both", padx=8, pady=10)

        button(parent, "+ Elegir imagen", self.choose_image, RED, WHITE).pack(
            fill="x", padx=13, pady=(0, 7)
        )
        self.analyze_button = button(
            parent, "Analizar imagen", self.analyze, GREEN, BG
        )
        self.analyze_button.pack(fill="x", padx=13)
        self.filename = label(
            parent, "Ninguna imagen seleccionada", 9, MUTED,
            wraplength=310, justify="left",
        )
        self.filename.pack(anchor="w", padx=14, pady=(10, 16))

    def build_features_panel(self, parent):
        thumbs = tk.Frame(parent, bg=PANEL)
        thumbs.pack(fill="x", padx=8)
        self.process_views = {}
        for name, title in (
            ("original", "ORIGINAL"), ("gray", "GRISES"), ("binary", "BINARIA")
        ):
            tile = tk.Frame(
                thumbs, bg=INNER, highlightbackground=BORDER, highlightthickness=1
            )
            tile.pack(side="left", expand=True, fill="both", padx=3)
            view = label(tile, "Sin imagen", 8, MUTED, bg=INNER)
            view.pack(expand=True, fill="both", pady=6)
            label(tile, title, 8, MUTED, True, INNER).pack(pady=(0, 6))
            self.process_views[name] = view

        label(parent, "BOX COUNTING  /  HOG  /  LBP  /  COLOR", 10, GREEN, True).pack(
            anchor="w", padx=14, pady=(19, 4)
        )
        self.graph = tk.Canvas(
            parent, bg=INNER, highlightthickness=1, highlightbackground=BORDER,
            height=240,
        )
        self.graph.pack(fill="both", expand=True, padx=13, pady=8)
        self.graph.bind("<Configure>", lambda _event: self.draw_plot())
        self.feature_label = label(
            parent, "BC: 8  |  HOG: 16200  |  LBP: 10  |  HSV: 144\nTotal: 16362 características",
            10, GREEN, True,
        )
        self.feature_label.pack(pady=(6, 3))
        self.dimension_label = label(
            parent, "Dimensiones fractales pendientes", 9, MUTED
        )
        self.dimension_label.pack(pady=(2, 4))
        self.processing_label = label(parent, "Procesamiento por determinar", 9, MUTED)
        self.processing_label.pack(pady=(0, 13))

    def build_result_panel(self, parent):
        preview = tk.Frame(parent, bg=INNER, height=170)
        preview.pack(fill="x", padx=13)
        preview.pack_propagate(False)
        self.result_preview = label(
            preview, "Esperando reconocimiento...", 10, MUTED, bg=INNER
        )
        self.result_preview.pack(expand=True, fill="both")

        label(parent, "RESULTADO DE KICKSMIND", 9, YELLOW, True).pack(
            anchor="w", padx=14, pady=(17, 5)
        )
        self.prediction_label = label(
            parent, "Sin clasificación", 19, WHITE, True, wraplength=310, anchor="w"
        )
        self.prediction_label.pack(anchor="w", padx=14)
        label(parent, "Algoritmo utilizado", 9, MUTED).pack(
            anchor="w", padx=14, pady=(15, 4)
        )
        self.algorithm_label = label(parent, "Se mostrará el modelo cargado", 10, GREEN, True)
        self.algorithm_label.pack(anchor="w", padx=14)

        label(parent, "PUNTUACIONES POR CATEGORÍA", 9, WHITE, True).pack(
            anchor="w", padx=14, pady=(22, 9)
        )
        self.probability_box = tk.Frame(parent, bg=PANEL)
        self.probability_box.pack(fill="x", padx=13)
        label(
            parent,
            "Los porcentajes son estimaciones del modelo;\nno garantizan que la predicción sea correcta.",
            8, MUTED, justify="left",
        ).pack(anchor="w", padx=14, pady=(12, 6))

    def display(self, target: tk.Label, image: Image.Image, key: str, size):
        copy = image.copy()
        copy.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(copy)
        self.photos[key] = photo
        target.configure(image=photo, text="")

    def clear_target(self, target: tk.Label, text: str):
        target.configure(image="", text=text)

    def choose_image(self):
        name = filedialog.askopenfilename(
            title="Seleccionar imagen del sneaker",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.jfif"),
                       ("Todos los archivos", "*.*")],
        )
        if not name:
            return
        try:
            with Image.open(name) as opened:
                original = opened.convert("RGB")
            self.image_path = Path(name)
            self.display(self.preview, original, "preview", (315, 340))
            self.filename.configure(text=self.image_path.name)
            self.clear_target(self.result_preview, "Esperando reconocimiento...")
            self.prediction_label.configure(text="Sin clasificación", fg=WHITE)
            for view in self.process_views.values():
                self.clear_target(view, "Sin imagen")
            self.counts = None
            self.draw_plot()
            self.dimension_label.configure(text="Dimensiones fractales pendientes")
            self.processing_label.configure(text="Procesamiento por determinar")
            for child in self.probability_box.winfo_children():
                child.destroy()
            self.update_status("Imagen seleccionada correctamente", GREEN)
        except Exception as exc:
            messagebox.showerror("Imagen no válida", str(exc))

    def load_model(self):
        if self.model is None:
            model_path = find_model()
            if not model_path.is_file():
                raise FileNotFoundError(
                    "No se encontró el modelo V3 ni el V2 de respaldo.\n"
                    "Ejecuta primero: py entrenar.py\n"
                    f"Ubicación buscada: {model_path}"
                )
            # Solo abrir modelos .joblib creados por ti o una fuente confiable.
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and loaded.get("version") == 3:
                model = loaded["model"]
                extractor = loaded["extractor"]
                algorithm = loaded["algorithm"]
                expected = FEATURE_COUNT_V3 if extractor == "v3" else FEATURE_COUNT_V2
                if loaded.get("feature_count") != expected:
                    raise ValueError("El paquete del modelo indica un extractor incompatible.")
            elif not isinstance(loaded, dict):
                model = loaded
                extractor = "v2"
                algorithm = "Regresión logística V2 (respaldo)"
                expected = FEATURE_COUNT_V2
            else:
                raise ValueError("Archivo de modelo desconocido o incompatible.")
            if not all(hasattr(model, attribute) for attribute in
                       ("predict", "predict_proba", "classes_", "n_features_in_")):
                raise ValueError("El archivo no contiene un clasificador compatible.")
            if model.n_features_in_ != expected:
                raise ValueError(f"Modelo incompatible: {model.n_features_in_} columnas; se esperaban {expected}.")
            self.model = model
            self.extractor = extractor
            self.algorithm = algorithm
        return self.model

    def analyze(self):
        if self.image_path is None:
            messagebox.showwarning("KicksMind", "Primero selecciona una imagen.")
            return
        self.analyze_button.configure(state="disabled")
        self.root.configure(cursor="watch")
        self.update_status("Analizando imagen...", YELLOW)
        self.root.update_idletasks()
        try:
            model = self.load_model()
            result, vector = (extract_v3(self.image_path) if self.extractor == "v3"
                              else extract_v2(self.image_path))
            prediction = str(model.predict(vector)[0]).replace("_", " ")
            votes = model.predict_proba(vector)[0]

            with Image.open(self.image_path) as opened:
                original = opened.convert("RGB")
            self.display(self.process_views["original"], original, "original", (93, 115))
            self.display(
                self.process_views["gray"], Image.fromarray(result.gray),
                "gray", (93, 115),
            )
            self.display(
                self.process_views["binary"], Image.fromarray(result.binary),
                "binary", (93, 115),
            )
            self.display(self.result_preview, original, "result", (290, 155))
            ranked = sorted(votes, reverse=True)
            uncertain = float(ranked[0]) < 0.55 or float(ranked[0] - ranked[1]) < 0.12
            self.prediction_label.configure(
                text=prediction + (" (revisar)" if uncertain else ""),
                fg=YELLOW if uncertain else GREEN,
            )
            self.algorithm_label.configure(text=f"{self.algorithm} · {self.extractor.upper()} · HOG + LBP + BC + HSV")
            self.processing_label.configure(text=(result.processing_mode if self.extractor == "v3"
                                                   else "Extractor original V2"))
            self.dimension_label.configure(
                text=(f"Dimensión de bordes: {result.edge_dimension:.3f}\n"
                      f"Dimensión de silueta: {result.silhouette_dimension:.3f}")
            )
            self.counts = result.box_counts
            self.draw_plot()
            self.show_votes(model.classes_, votes)
            self.update_status(
                "Predicción incierta: considera otra foto del sneaker" if uncertain else
                "Análisis completado · resultado orientativo",
                YELLOW if uncertain else GREEN,
            )
        except Exception as exc:
            self.update_status("No se pudo completar el análisis", RED)
            messagebox.showerror("Error en KicksMind", str(exc))
        finally:
            self.root.configure(cursor="")
            self.analyze_button.configure(state="normal")

    def show_votes(self, classes, votes):
        for child in self.probability_box.winfo_children():
            child.destroy()
        for class_name, vote in zip(classes, votes):
            row = tk.Frame(self.probability_box, bg=PANEL)
            row.pack(fill="x", pady=4)
            label(row, str(class_name).replace("_", " "), 9, WHITE, width=16,
                  anchor="w").pack(side="left")
            bar = ttk.Progressbar(
                row, maximum=100, value=float(vote * 100),
                style="Kicks.Horizontal.TProgressbar", length=88,
            )
            bar.pack(side="left", fill="x", expand=True, padx=6)
            label(row, f"{vote * 100:.1f}%", 9, GREEN, True, width=6,
                  anchor="e").pack(side="right")

    def draw_plot(self):
        if not hasattr(self, "graph") or not self.graph.winfo_exists():
            return
        c = self.graph
        c.delete("all")
        width = max(c.winfo_width(), 270)
        height = max(c.winfo_height(), 180)
        if self.counts is None:
            c.create_text(
                width / 2, height / 2,
                text="La gráfica aparecerá\ndespués del análisis",
                fill=MUTED, font=(FONT, 10), justify="center",
            )
            return

        left, right, top, bottom = 49, width - 18, 23, height - 40
        if right <= left or bottom <= top:
            return
        maximum = max(max(self.counts), 1)
        for i in range(5):
            y = bottom - (bottom - top) * i / 4
            value = round(maximum * i / 4)
            c.create_line(left, y, right, y, fill=BORDER)
            c.create_text(left - 6, y, text=str(value), fill=MUTED,
                          font=(FONT, 8), anchor="e")
        c.create_text(width / 2, 10, text="Cajas ocupadas por tamaño",
                      fill=WHITE, font=(FONT, 9, "bold"))
        points = []
        for i, (size, value) in enumerate(zip(BOX_SIZES, self.counts)):
            x = left + (right - left) * i / (len(BOX_SIZES) - 1)
            y = bottom - (bottom - top) * value / maximum
            c.create_text(x, bottom + 15, text=str(size), fill=MUTED, font=(FONT, 8))
            points.extend((x, y))
        if len(points) >= 4:
            c.create_line(*points, fill=GREEN, width=2, smooth=False)
        for x, y in zip(points[::2], points[1::2]):
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=GREEN, outline=INNER)
        c.create_text(width / 2, height - 9, text="Tamaño de caja",
                      fill=MUTED, font=(FONT, 8))

    def update_status(self, text, color=MUTED):
        self.status.configure(text=text, fg=color)
        self.root.update_idletasks()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    KicksMind().run()
