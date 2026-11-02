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

def graficar_distribuciones_categoricas(df, columnas=None, guardar_path=None):
    """
    Genera gráficos de barras con frecuencias relativas (%) para variables categóricas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset con los datos de la planta.
    columnas : list, optional
        Lista de variables categóricas a graficar.
    guardar_path : str, optional
        Ruta para guardar la figura generada.
    """
    if columnas is None:
        columnas = ["WeekStatus", "Day_of_week", "Load_Type"]
        
    # Filtrar solo columnas existentes
    columnas = [col for col in columnas if col in df.columns]
    
    fig, axes = plt.subplots(1, len(columnas), figsize=(6 * len(columnas), 5))
    if len(columnas) == 1:
        axes = [axes]
        
    dias_orden = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    
    for ax, col in zip(axes, columnas):
        # Calcular proporciones
        if col == "Day_of_week":
            orden = [d for d in dias_orden if d in df[col].unique()]
            props = (df[col].value_counts(normalize=True).reindex(orden) * 100)
        else:
            props = (df[col].value_counts(normalize=True) * 100)
            
        # Graficar barras
        sns.barplot(x=props.index, y=props.values, ax=ax, palette="Blues_d")
        
        # Formato visual
        ax.set_title(f"Distribución de {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Proporción (%)", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=30 if col == "Day_of_week" else 0)
        
        # Agregar etiquetas de porcentaje sobre cada barra
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=9, color='black',
                            xytext=(0, 3), textcoords='offset points')
                            
    plt.tight_layout()
    
    if guardar_path:
        plt.savefig(guardar_path, dpi=300, bbox_inches='tight')
        print(f" Gráfico guardado exitosamente en: {guardar_path}")
        
    plt.show()