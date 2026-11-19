"""Gráficos de la Tarea 5: frontera de decisión (hiperplano) y comparaciones de SVM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from modelos_aprendizaje import pronostico as pr
from . import modelos_svm as ms

COLOR_NORMAL = "#4C78A8"
COLOR_ALERTA = "#E45756"
COLORES_KERNEL = {"lineal": "#4C78A8", "polinomial": "#F58518", "rbf": "#54A24B"}


def graficar_frontera_2d(X, y, parametros: dict | None = None, n_puntos: int = 1500,
                         semilla: int = ms.SEMILLA, class_weight="balanced",
                         disposicion: str = "vertical"):
    """Hiperplano de los tres kernels en un plano de 2 variables.

    Variables del gráfico: consumo actual (kWh) y hora del instante pronosticado.
    El SVM se reentrena SOLO con estas dos variables para poder dibujarlo; sirve
    para ver cómo cambia la forma de la frontera según el kernel, no es el modelo
    final (que usa las 13 variables).

    - Fondo: región que el modelo clasifica como alerta (rojo) o normal (azul).
    - Línea negra continua: el hiperplano (f(x) = 0). Líneas punteadas: los márgenes.
    - Círculos huecos: vectores de soporte (los puntos que definen la frontera).

    disposicion='vertical' apila los tres paneles (se lee mejor al exportar a PDF);
    'horizontal' los pone en una fila. Los vectores de soporte del título corresponden
    a estos modelos de 2 variables con `n_puntos` puntos, no al modelo de 13 variables.
    """
    parametros = parametros or {}
    d = pd.DataFrame({"Consumo_t": X["Consumo_t"],
                      "Hora_destino": pr.hora_destino(X)}, index=X.index)
    rng = np.random.default_rng(semilla)
    idx = np.sort(rng.choice(len(d), size=min(n_puntos, len(d)), replace=False))
    d, yy = d.iloc[idx], y.iloc[idx]

    x_min, x_max = d["Consumo_t"].min() - 2, d["Consumo_t"].max() + 2
    h_min, h_max = 0, 24
    gx, gh = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(h_min, h_max, 300))
    malla = pd.DataFrame({"Consumo_t": gx.ravel(), "Hora_destino": gh.ravel()})

    if disposicion == "vertical":
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 13.0), sharex=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, nombre in zip(axes, ms.KERNELS):
        modelo = ms.crear_svm(nombre, class_weight=class_weight, **parametros.get(nombre, {}))
        modelo.fit(d, yy)
        z = modelo.decision_function(malla).reshape(gx.shape)
        ax.contourf(gx, gh, z > 0, levels=[-0.5, 0.5, 1.5],
                    colors=[COLOR_NORMAL, COLOR_ALERTA], alpha=0.15)
        ax.contour(gx, gh, z, levels=[-1, 1], colors="k", linestyles=":", linewidths=0.8)
        ax.contour(gx, gh, z, levels=[0], colors="k", linewidths=1.8)
        ax.scatter(d.loc[yy == 0, "Consumo_t"], d.loc[yy == 0, "Hora_destino"], s=6,
                   c=COLOR_NORMAL, alpha=0.45, label="Normal")
        ax.scatter(d.loc[yy == 1, "Consumo_t"], d.loc[yy == 1, "Hora_destino"], s=6,
                   c=COLOR_ALERTA, alpha=0.55, label="Alerta (carga máxima)")
        sv = modelo.named_steps["svm"].support_
        ax.scatter(d["Consumo_t"].iloc[sv], d["Hora_destino"].iloc[sv], s=34,
                   facecolors="none", edgecolors="k", linewidths=0.5, label="Vectores de soporte")
        ax.set_title(f"Kernel {nombre}: {len(sv)} vectores de soporte de {len(d):,} puntos".replace(",", "."))
        ax.set_xlabel("Consumo actual (kWh)")
        ax.set_ylabel("Hora del instante pronosticado")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(h_min, h_max)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Hiperplano de decisión según el kernel (SVM con 2 variables)", y=1.0)
    fig.tight_layout()
    return fig


def graficar_metricas_cv(folds: dict, metrica: str = "F1", titulo: str | None = None):
    """Boxplot de una métrica por kernel, con un punto por cada fold."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    nombres = list(folds)
    datos = [folds[n][metrica].to_numpy() for n in nombres]
    ax.boxplot(datos, widths=0.5, showmeans=True)
    ax.set_xticks(range(1, len(nombres) + 1), nombres)
    for i, d in enumerate(datos, start=1):
        ax.scatter(np.full(len(d), i), d, s=26, color=COLORES_KERNEL.get(nombres[i - 1], "gray"),
                   zorder=3)
    ax.set_ylabel(metrica)
    ax.set_title(titulo or f"{metrica} por fold según el kernel")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[
        Line2D([], [], color="tab:orange", lw=1.5, label="Mediana de los folds"),
        Line2D([], [], color="tab:green", marker="^", ls="", label="Promedio de los folds"),
        Line2D([], [], color="gray", marker="o", ls="", label="Cada fold"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def graficar_comparacion(tabla: pd.DataFrame, metricas=("Recall", "Precision", "F1"),
                         titulo: str = "Desempeño en prueba por kernel"):
    """Barras agrupadas de las métricas clave (filas = kernels)."""
    ax = tabla[list(metricas)].plot.bar(figsize=(7.5, 4.2), width=0.75, rot=0)
    ax.set_ylim(0, 1)
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def graficar_matrices_confusion(y_real, preds: dict):
    """Una matriz de confusión por modelo (kernel), con conteos."""
    fig, axes = plt.subplots(1, len(preds), figsize=(4.4 * len(preds), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (nombre, p) in zip(axes, preds.items()):
        cm = ms.matriz_confusion(y_real, p).to_numpy()
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Normal", "Alerta"])
        ax.set_yticks([0, 1], ["Normal", "Alerta"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(nombre)
    fig.tight_layout()
    return fig


def graficar_curva_aprendizaje(r: pd.DataFrame, metrica: str = "F1"):
    """F1 (o AUC) de cada kernel según el número de casos de entrenamiento.

    La línea es el promedio de las repeticiones y la banda, ± 1 desviación estándar.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for nombre in ms.KERNELS:
        d = r[r["Kernel"] == nombre].groupby("n")[metrica].agg(["mean", "std"]).fillna(0)
        ax.plot(d.index, d["mean"], marker="o", label=nombre, color=COLORES_KERNEL[nombre])
        ax.fill_between(d.index, d["mean"] - d["std"], d["mean"] + d["std"], alpha=0.15,
                        color=COLORES_KERNEL[nombre])
    ax.set_xscale("log")
    ax.set_xlabel("Casos de entrenamiento (escala logarítmica)")
    ax.set_ylabel(metrica)
    ax.set_title(f"Curva de aprendizaje: {metrica} en el bloque de validación")
    ax.grid(alpha=0.3)
    ax.legend(title="Kernel")
    fig.tight_layout()
    return fig


def graficar_curva_validacion_gamma(t: pd.DataFrame):
    """Puntaje de entrenamiento y de validación del RBF según gamma (un panel por métrica)."""
    metricas = list(t["Metrica"].unique())
    fig, axes = plt.subplots(1, len(metricas), figsize=(6.2 * len(metricas), 4.2))
    axes = np.atleast_1d(axes)
    for ax, met in zip(axes, metricas):
        d = t[t["Metrica"] == met]
        ax.plot(d["gamma"], d["Entrenamiento"], marker="o", color="#4C78A8", label="Entrenamiento")
        ax.plot(d["gamma"], d["Validacion"], marker="o", color="#E45756", label="Validación (CV temporal)")
        ax.set_xscale("log")
        ax.set_xlabel("gamma (escala logarítmica)")
        ax.set_ylabel(met)
        ax.set_title(f"RBF: {met} según gamma")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig
