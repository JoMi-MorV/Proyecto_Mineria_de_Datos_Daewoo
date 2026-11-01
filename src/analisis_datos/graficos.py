"""Módulo para la visualización de datos y diagnóstico de distribuciones.

Proyecto Daewoo Steel Co.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def graficar_distribucion_y_boxplots(
    df: pd.DataFrame, variables: list[str], guardar: bool = False
) -> None:
    """Genera histogramas y boxplots para visualizar asimetría y outliers."""
    sns.set_theme(style="whitegrid")

    for var in variables:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # 1. Histograma + KDE
        sns.histplot(
            df[var], kde=True, ax=axes[0], color="skyblue", edgecolor="black"
        )
        axes[0].set_title(f"Distribución e Histograma: {var}", fontsize=12)
        axes[0].set_xlabel(var)
        axes[0].set_ylabel("Frecuencia")

        # 2. Diagrama de Caja (Boxplot)
        sns.boxplot(x=df[var], ax=axes[1], color="salmon")
        axes[1].set_title(f"Diagrama de Caja (Outliers): {var}", fontsize=12)
        axes[1].set_xlabel(var)

        plt.tight_layout()

        if guardar:
            ruta_img = Path(f"../notebooks/imagenes/{var}_distribucion.png")
            ruta_img.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_img, dpi=300)

        plt.show()