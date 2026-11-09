
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
import joblib

from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from box_counting import box_counting, recortar_y_normalizar
from hog_lbp import extraer_hog, extraer_lbp
from preprocesamiento import preprocesar


# ==================================================
# 1. CONFIGURACION GENERAL
# ==================================================

COLOR_FONDO = "#080d13"
COLOR_PANEL = "#111923"
COLOR_BORDE = "#263443"
COLOR_VERDE = "#a3ff12"
COLOR_ROJO = "#ff4055"
COLOR_TEXTO = "#f1f5f9"
COLOR_SECUNDARIO = "#94a3b8"
COLOR_AMARILLO = "#fbbf24"

ARCHIVO_MODELO = "modelo_final_kicksmind.joblib"

TAMANOS_CAJA = [2, 4, 8, 16, 32, 64]

ANCHO = 1100
ALTO = 760


# ==================================================
# 2. VARIABLES GLOBALES
# ==================================================

ruta_imagen = None

modelo_kicksmind = None

imagenes_vistas = {}

imagen_original_tk = None
imagen_resultado_tk = None

figura = None
ax = None
canvas = None


# ==================================================
# 3. VENTANA PRINCIPAL
# ==================================================

ventana = tk.Tk()

ventana.title("KicksMind | Sneaker Recognition")

ventana.geometry(f"{ANCHO}x{ALTO}")
ventana.minsize(900, 650)

ventana.configure(bg=COLOR_FONDO)


# ==================================================
# 4. FUNCIONES AUXILIARES
# ==================================================

def crear_titulo(parent, texto):

    return tk.Label(
        parent,
        text=texto,
        font=("Arial", 11, "bold"),
        fg=COLOR_TEXTO,
        bg=COLOR_PANEL
    )


def limpiar_vista(etiqueta, texto):

    etiqueta.configure(
        image="",
        text=texto,
        fg=COLOR_SECUNDARIO
    )


def mostrar_imagen(imagen, etiqueta, nombre, tamano=(120, 105)):

    imagen_pil = Image.fromarray(imagen)

    imagen_pil.thumbnail(tamano)

    imagen_tk = ImageTk.PhotoImage(imagen_pil)

    # Conservar la referencia de la imagen
    imagenes_vistas[nombre] = imagen_tk

    etiqueta.configure(
        image=imagen_tk,
        text=""
    )


def actualizar_estado(texto, color=COLOR_SECUNDARIO):

    etiqueta_estado.configure(
        text=texto,
        fg=color
    )

    ventana.update_idletasks()


# ==================================================
# 5. EXTRAER CARACTERISTICAS BOX COUNTING
# ==================================================

def extraer_caracteristicas_bc(ruta):

    # Procesar la imagen
    gris, binaria, bordes = preprocesar(ruta)

    # Box Counting sobre los bordes
    d_b, N_b = box_counting(
        recortar_y_normalizar(bordes)
    )

    # Box Counting sobre la silueta
    d_s, _ = box_counting(
        recortar_y_normalizar(binaria)
    )

    # Construir las 8 características
    caracteristicas = [d_b, d_s] + list(
        np.log(np.maximum(N_b, 1))
    )

    return (
        np.array(caracteristicas),
        gris,
        binaria,
        bordes,
        N_b,
        d_b,
        d_s
    )


# ==================================================
# 6. EXTRAER CARACTERISTICAS COMBINADAS
# ==================================================

def extraer_caracteristicas_combinadas(ruta):

    # Obtener características Box Counting
    bc, gris, binaria, bordes, N_b, d_b, d_s = (
        extraer_caracteristicas_bc(ruta)
    )

    # Obtener características HOG
    hog = extraer_hog(ruta)

    # Obtener características LBP
    lbp = extraer_lbp(ruta)

    # Combinar en el mismo orden del entrenamiento
    combinadas = np.concatenate([
        bc,
        hog,
        lbp
    ])

    # Convertir a matriz de una imagen
    combinadas = combinadas.reshape(1, -1)

    return (
        combinadas,
        gris,
        binaria,
        bordes,
        N_b,
        d_b,
        d_s
    )


# ==================================================
# 7. CARGAR EL MODELO
# ==================================================

def cargar_modelo():

    global modelo_kicksmind

    if modelo_kicksmind is None:

        ruta_modelo = Path(ARCHIVO_MODELO)

        if not ruta_modelo.exists():

            raise FileNotFoundError(
                f"No se encontró el modelo: {ARCHIVO_MODELO}"
            )

        modelo_kicksmind = joblib.load(ruta_modelo)

        # Comprobar que sea el modelo esperado
        if not hasattr(modelo_kicksmind, "n_features_in_"):

            raise ValueError(
                "El archivo no contiene un clasificador compatible."
            )

        if modelo_kicksmind.n_features_in_ != 8118:

            raise ValueError(
                "El modelo no tiene las 8118 características esperadas."
            )

    return modelo_kicksmind


# ==================================================
# 8. SELECCIONAR IMAGEN
# ==================================================

def seleccionar_imagen():

    global ruta_imagen
    global imagen_original_tk
    global imagen_resultado_tk

    archivo = filedialog.askopenfilename(
        title="Seleccionar fotografía del sneaker",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.webp")
        ]
    )

    if not archivo:
        return

    try:

        imagen = Image.open(archivo).convert("RGB")

        imagen.thumbnail((300, 270))

        imagen_original_tk = ImageTk.PhotoImage(imagen)

        etiqueta_imagen.configure(
            image=imagen_original_tk,
            text=""
        )

        ruta_imagen = archivo

        etiqueta_archivo.configure(
            text=Path(archivo).name
        )

        # Limpiar resultados anteriores
        etiqueta_resultado.configure(
            text="Pendiente de análisis",
            fg=COLOR_TEXTO
        )

        limpiar_vista(
            etiqueta_foto_resultado,
            "Esperando reconocimiento..."
        )

        imagen_resultado_tk = None

        for nombre in ["grises", "binaria"]:
            imagenes_vistas.pop(nombre, None)

        limpiar_vista(vista_grises, "Escala de grises")
        limpiar_vista(vista_binaria, "Imagen binaria")

        limpiar_probabilidades()

        ax.clear()
        ax.set_facecolor(COLOR_FONDO)
        canvas.draw()

        actualizar_estado(
            "Imagen seleccionada correctamente",
            COLOR_VERDE
        )

    except Exception as error:

        messagebox.showerror(
            "Error al abrir imagen",
            str(error)
        )


# ==================================================
# 9. MOSTRAR PROBABILIDADES
# ==================================================

def limpiar_probabilidades():

    for widget in marco_probabilidades.winfo_children():

        widget.destroy()


def mostrar_probabilidades(clasificador, probabilidades):

    limpiar_probabilidades()

    tk.Label(
        marco_probabilidades,
        text="DISTRIBUCIÓN DE VOTOS POR CATEGORÍA",
        font=("Arial", 9, "bold"),
        fg=COLOR_TEXTO,
        bg=COLOR_PANEL
    ).pack(
        anchor="w",
        pady=(3, 8)
    )

    for clase, probabilidad in zip(
        clasificador.classes_,
        probabilidades
    ):

        fila = tk.Frame(
            marco_probabilidades,
            bg=COLOR_PANEL
        )

        fila.pack(
            fill="x",
            pady=3
        )

        nombre = clase.replace("_", " ")

        tk.Label(
            fila,
            text=nombre,
            font=("Arial", 8),
            fg=COLOR_TEXTO,
            bg=COLOR_PANEL,
            width=16,
            anchor="w"
        ).pack(side="left")

        barra = ttk.Progressbar(
            fila,
            maximum=100,
            value=float(probabilidad * 100),
            length=105
        )

        barra.pack(
            side="left",
            padx=5
        )

        tk.Label(
            fila,
            text=f"{probabilidad * 100:.1f}%",
            font=("Arial", 8, "bold"),
            fg=COLOR_VERDE,
            bg=COLOR_PANEL,
            width=7
        ).pack(side="left")


# ==================================================
# 10. ACTUALIZAR GRAFICA BOX COUNTING
# ==================================================

def actualizar_grafica(N_b):

    ax.clear()

    ax.plot(
        TAMANOS_CAJA,
        N_b,
        marker="o",
        markersize=5,
        linewidth=2,
        color=COLOR_VERDE
    )

    ax.set_title(
        "Cajas ocupadas por tamaño",
        color=COLOR_TEXTO,
        fontsize=9
    )

    ax.set_xlabel(
        "Tamaño de caja",
        color=COLOR_SECUNDARIO,
        fontsize=8
    )

    ax.set_ylabel(
        "Cajas ocupadas",
        color=COLOR_SECUNDARIO,
        fontsize=8
    )

    ax.set_xscale(
        "log",
        base=2
    )

    ax.set_xticks(TAMANOS_CAJA)

    ax.set_xticklabels(
        TAMANOS_CAJA,
        color=COLOR_SECUNDARIO,
        fontsize=7
    )

    ax.tick_params(
        axis="y",
        colors=COLOR_SECUNDARIO,
        labelsize=7
    )

    ax.set_facecolor(COLOR_FONDO)

    ax.grid(
        True,
        color=COLOR_BORDE,
        alpha=0.8
    )

    figura.patch.set_facecolor(COLOR_FONDO)

    figura.tight_layout()

    canvas.draw()


# ==================================================
# 11. ANALIZAR IMAGEN
# ==================================================

def analizar_imagen():

    global imagen_resultado_tk

    if ruta_imagen is None:

        messagebox.showwarning(
            "KicksMind",
            "Primero selecciona una imagen."
        )

        return

    try:

        actualizar_estado(
            "Procesando imagen...",
            COLOR_AMARILLO
        )

        ventana.update()

        # Cargar modelo
        clasificador = cargar_modelo()

        # Extraer las características
        (
            caracteristicas,
            gris,
            binaria,
            bordes,
            N_b,
            d_b,
            d_s
        ) = extraer_caracteristicas_combinadas(
            ruta_imagen
        )

        # Comprobar dimensiones
        if caracteristicas.shape[1] != clasificador.n_features_in_:

            raise ValueError(
                "Las características de la imagen no coinciden con el modelo."
            )

        # Realizar predicción
        prediccion = clasificador.predict(
            caracteristicas
        )[0]

        # Obtener distribución de votos
        probabilidades = clasificador.predict_proba(
            caracteristicas
        )[0]

        # Mostrar imágenes procesadas
        mostrar_imagen(
            gris,
            vista_grises,
            "grises"
        )

        mostrar_imagen(
            binaria,
            vista_binaria,
            "binaria"
        )

        # Mostrar fotografía en el resultado
        imagen = Image.open(ruta_imagen).convert("RGB")

        imagen.thumbnail((280, 145))

        imagen_resultado_tk = ImageTk.PhotoImage(imagen)

        etiqueta_foto_resultado.configure(
            image=imagen_resultado_tk,
            text=""
        )

        # Mostrar nombre del sneaker
        etiqueta_resultado.configure(
            text=prediccion.replace("_", " "),
            fg=COLOR_VERDE
        )

        # Mostrar probabilidades
        mostrar_probabilidades(
            clasificador,
            probabilidades
        )

        # Actualizar gráfica
        actualizar_grafica(N_b)

        # Mostrar dimensiones
        etiqueta_caracteristicas.configure(
            text=(
                f"BC: 8  |  HOG: 8100  |  LBP: 10\n"
                f"Total: {caracteristicas.shape[1]} características"
            )
        )

        # Mostrar dimensiones fractales
        etiqueta_dimensiones.configure(
            text=(
                f"Dimensión de bordes: {d_b:.3f}\n"
                f"Dimensión de silueta: {d_s:.3f}"
            )
        )

        actualizar_estado(
            "Análisis completado correctamente",
            COLOR_VERDE
        )

    except Exception as error:

        actualizar_estado(
            "Ocurrió un error durante el análisis",
            COLOR_ROJO
        )

        messagebox.showerror(
            "Error en KicksMind",
            str(error)
        )


# ==================================================
# 12. ENCABEZADO
# ==================================================

encabezado = tk.Frame(
    ventana,
    bg=COLOR_FONDO
)

encabezado.pack(
    fill="x",
    padx=22,
    pady=(16, 10)
)

tk.Label(
    encabezado,
    text="KICKSMIND",
    font=("Arial", 25, "bold"),
    fg=COLOR_VERDE,
    bg=COLOR_FONDO
).pack(side="left")

tk.Label(
    encabezado,
    text="SNEAKER RECOGNITION SYSTEM",
    font=("Arial", 9, "bold"),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_FONDO
).pack(
    side="left",
    padx=15,
    pady=10
)

tk.Label(
    encabezado,
    text="VERSIÓN FUNCIONAL",
    font=("Arial", 8, "bold"),
    fg=COLOR_VERDE,
    bg=COLOR_PANEL,
    padx=12,
    pady=8
).pack(side="right")


# ==================================================
# 13. TITULO DEL PANEL
# ==================================================

tk.Label(
    ventana,
    text="Panel de análisis",
    font=("Arial", 19, "bold"),
    fg=COLOR_TEXTO,
    bg=COLOR_FONDO
).pack(
    anchor="w",
    padx=22
)

tk.Label(
    ventana,
    text="Imagen  →  procesamiento  →  características  →  clasificación",
    font=("Arial", 9),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_FONDO
).pack(
    anchor="w",
    padx=22,
    pady=(3, 12)
)


# ==================================================
# 14. CONTENEDOR PRINCIPAL
# ==================================================

contenedor = tk.Frame(
    ventana,
    bg=COLOR_FONDO
)

contenedor.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=5
)

contenedor.columnconfigure(0, weight=34)
contenedor.columnconfigure(1, weight=33)
contenedor.columnconfigure(2, weight=33)

contenedor.rowconfigure(0, weight=1)


# ==================================================
# 15. PANEL 1 - SELECCION DE IMAGEN
# ==================================================

panel1 = tk.Frame(
    contenedor,
    bg=COLOR_PANEL,
    highlightbackground=COLOR_BORDE,
    highlightthickness=1
)

panel1.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=4
)

crear_titulo(
    panel1,
    "01   Selecciona una imagen"
).pack(
    anchor="w",
    padx=13,
    pady=15
)


marco_imagen = tk.Frame(
    panel1,
    bg=COLOR_FONDO,
    highlightbackground=COLOR_BORDE,
    highlightthickness=1
)

marco_imagen.pack(
    fill="both",
    expand=True,
    padx=12,
    pady=5
)

etiqueta_imagen = tk.Label(
    marco_imagen,
    text="Selecciona una fotografía\npara comenzar",
    font=("Arial", 11),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_FONDO
)

etiqueta_imagen.pack(
    expand=True,
    fill="both"
)


botones = tk.Frame(
    panel1,
    bg=COLOR_PANEL
)

botones.pack(
    fill="x",
    padx=12,
    pady=10
)


tk.Button(
    botones,
    text="+ Elegir imagen",
    command=seleccionar_imagen,
    bg=COLOR_ROJO,
    fg="white",
    activebackground="#dc3045",
    activeforeground="white",
    font=("Arial", 9, "bold"),
    relief="flat",
    padx=10,
    pady=10,
    cursor="hand2"
).pack(
    side="left",
    padx=(0, 6)
)


tk.Button(
    botones,
    text="Analizar imagen ↗",
    command=analizar_imagen,
    bg="#202d3b",
    fg=COLOR_TEXTO,
    activebackground="#314256",
    activeforeground="white",
    font=("Arial", 9, "bold"),
    relief="flat",
    padx=10,
    pady=10,
    cursor="hand2"
).pack(side="left")


etiqueta_archivo = tk.Label(
    panel1,
    text="Ninguna imagen seleccionada",
    font=("Arial", 8),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_PANEL
)

etiqueta_archivo.pack(
    anchor="w",
    padx=14,
    pady=(0, 10)
)


# ==================================================
# 16. PANEL 2 - PROCESAMIENTO
# ==================================================

panel2 = tk.Frame(
    contenedor,
    bg=COLOR_PANEL,
    highlightbackground=COLOR_BORDE,
    highlightthickness=1
)

panel2.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=4
)

crear_titulo(
    panel2,
    "02   Procesamiento + Box Counting"
).pack(
    anchor="w",
    padx=12,
    pady=15
)


marco_procesamiento = tk.Frame(
    panel2,
    bg=COLOR_PANEL
)

marco_procesamiento.pack(
    fill="x",
    padx=6
)


def crear_vista_proceso(texto):

    tarjeta = tk.Frame(
        marco_procesamiento,
        bg=COLOR_FONDO,
        highlightbackground=COLOR_BORDE,
        highlightthickness=1
    )

    tarjeta.pack(
        side="left",
        expand=True,
        fill="both",
        padx=3
    )

    vista = tk.Label(
        tarjeta,
        text="Sin imagen",
        bg=COLOR_FONDO,
        fg=COLOR_SECUNDARIO,
        height=5
    )

    vista.pack(
        fill="both",
        expand=True,
        padx=2,
        pady=3
    )

    tk.Label(
        tarjeta,
        text=texto,
        font=("Arial", 7, "bold"),
        fg=COLOR_SECUNDARIO,
        bg=COLOR_FONDO
    ).pack(pady=5)

    return vista


vista_original = crear_vista_proceso("ORIGINAL")
vista_grises = crear_vista_proceso("GRISES")
vista_binaria = crear_vista_proceso("BINARIA")


tk.Label(
    panel2,
    text="ANÁLISIS DE CARACTERÍSTICAS",
    font=("Arial", 9, "bold"),
    fg=COLOR_TEXTO,
    bg=COLOR_PANEL
).pack(
    anchor="w",
    padx=12,
    pady=(20, 7)
)


tk.Label(
    panel2,
    text="BOX COUNTING  |  HOG  |  LBP",
    font=("Arial", 10, "bold"),
    fg=COLOR_VERDE,
    bg=COLOR_PANEL
).pack(pady=5)


marco_grafica = tk.Frame(
    panel2,
    bg=COLOR_FONDO,
    highlightbackground=COLOR_BORDE,
    highlightthickness=1
)

marco_grafica.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=5
)


figura = Figure(
    figsize=(4, 2),
    dpi=85
)

ax = figura.add_subplot(111)

ax.set_facecolor(COLOR_FONDO)

figura.patch.set_facecolor(COLOR_FONDO)

canvas = FigureCanvasTkAgg(
    figura,
    master=marco_grafica
)

canvas.get_tk_widget().pack(
    fill="both",
    expand=True
)


etiqueta_caracteristicas = tk.Label(
    panel2,
    text="BC: 8  |  HOG: 8100  |  LBP: 10\nTotal: 8118 características",
    font=("Arial", 9, "bold"),
    fg=COLOR_VERDE,
    bg=COLOR_PANEL
)

etiqueta_caracteristicas.pack(
    pady=(8, 3)
)


etiqueta_dimensiones = tk.Label(
    panel2,
    text="Dimensiones fractales pendientes",
    font=("Arial", 8),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_PANEL
)

etiqueta_dimensiones.pack(
    pady=(3, 10)
)


# ==================================================
# 17. PANEL 3 - RESULTADO
# ==================================================

panel3 = tk.Frame(
    contenedor,
    bg=COLOR_PANEL,
    highlightbackground=COLOR_BORDE,
    highlightthickness=1
)

panel3.grid(
    row=0,
    column=2,
    sticky="nsew",
    padx=4
)

crear_titulo(
    panel3,
    "03   Resultado de clasificación"
).pack(
    anchor="w",
    padx=12,
    pady=15
)


marco_resultado = tk.Frame(
    panel3,
    bg=COLOR_FONDO,
    height=145
)

marco_resultado.pack(
    fill="x",
    padx=10
)

marco_resultado.pack_propagate(False)


etiqueta_foto_resultado = tk.Label(
    marco_resultado,
    text="Esperando reconocimiento...",
    font=("Arial", 10),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_FONDO
)

etiqueta_foto_resultado.pack(
    expand=True
)


tk.Label(
    panel3,
    text="RESULTADO DE KICKSMIND",
    font=("Arial", 8, "bold"),
    fg=COLOR_AMARILLO,
    bg=COLOR_PANEL
).pack(
    anchor="w",
    padx=12,
    pady=(13, 4)
)


etiqueta_resultado = tk.Label(
    panel3,
    text="Sin clasificación",
    font=("Arial", 17, "bold"),
    fg=COLOR_TEXTO,
    bg=COLOR_PANEL,
    wraplength=270
)

etiqueta_resultado.pack(
    anchor="w",
    padx=12,
    pady=5
)


tk.Label(
    panel3,
    text="Algoritmo utilizado",
    font=("Arial", 9),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_PANEL
).pack(
    anchor="w",
    padx=12,
    pady=(12, 4)
)


tk.Label(
    panel3,
    text="Random Forest · 300 árboles",
    font=("Arial", 10, "bold"),
    fg=COLOR_VERDE,
    bg=COLOR_PANEL
).pack(
    anchor="w",
    padx=12
)


marco_probabilidades = tk.Frame(
    panel3,
    bg=COLOR_PANEL
)

marco_probabilidades.pack(
    fill="x",
    padx=12,
    pady=12
)


# ==================================================
# 18. ESTILO DE BARRAS
# ==================================================

estilo = ttk.Style()

estilo.theme_use("clam")

estilo.configure(
    "Kicks.Horizontal.TProgressbar",
    troughcolor=COLOR_FONDO,
    background=COLOR_VERDE,
    bordercolor=COLOR_BORDE,
    lightcolor=COLOR_VERDE,
    darkcolor=COLOR_VERDE
)


# ==================================================
# 19. PIE DE PAGINA
# ==================================================

etiqueta_estado = tk.Label(
    ventana,
    text="Sistema listo para analizar",
    font=("Arial", 8, "bold"),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_FONDO
)

etiqueta_estado.pack(pady=(5, 2))


tk.Label(
    ventana,
    text="KICKSMIND  |  SNEAKER RECOGNITION SYSTEM",
    font=("Arial", 8),
    fg=COLOR_SECUNDARIO,
    bg=COLOR_FONDO
).pack(pady=(2, 8))


# ==================================================
# 20. INICIAR APLICACION
# ==================================================

ventana.mainloop()
