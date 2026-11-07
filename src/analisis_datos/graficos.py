"""Módulo para la visualización de datos y diagnóstico de distribuciones.

Proyecto Daewoo Steel Co.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def graficar_boxplots(df, variables_num, titulos, colores):
    """Genera un panel ampliado de subplots tipo boxplot para variables cuantitativas,

    asegurando alta visibilidad de los outliers estadísticos.
    """
    sns.set_theme(style='whitegrid', palette='muted')

    # Aumentamos el tamaño general para que cada gráfico sea amplio y claro (ancho: 16, alto: 22)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 22))
    axes = axes.flatten()

    for i, var in enumerate(variables_num):
        sns.boxplot(
            x=df[var],
            ax=axes[i],
            color=colores[i],
            fliersize=3.5,  # Tamaños de los puntos atípicos
            linewidth=1.3,
            flierprops=dict(
                marker='o',
                markerfacecolor='#d62728',  # Rojo destacado para visualización clara
                markeredgecolor='#800000',  # Borde oscuro para que no se pierdan
                alpha=0.6,  # Opacidad visible
            ),
        )

        axes[i].set_title(
            f'Boxplot: {titulos[i]}', fontsize=13, fontweight='bold', pad=10
        )
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=11)

    # Eliminar la octava celda libre del panel
    fig.delaxes(axes[7])

    plt.tight_layout(pad=3.0)

    plt.show()

def graficar_distribucion_individual(
    df: pd.DataFrame, variables: list[str] = None
) -> None:
    """
    Genera y muestra de manera INDIVIDUAL un gráfico de Histograma + KDE 
    para cada variable cuantitativa enviada en la lista.
    """
    sns.set_theme(style="whitegrid")

    # Si no se especifica lista, se procesan las 7 variables cuantitativas del proyecto
    if variables is None:
        variables = [
            "Usage_kWh",
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            "NSM"
        ]

    # Filtrar solo las columnas que existan en el DataFrame
    variables = [col for col in variables if col in df.columns]

    for var in variables:
        fig, ax = plt.subplots(figsize=(8, 5))

        # 1. Histograma (barras en celeste con borde negro)
        sns.histplot(
            df[var], kde=False, ax=ax, color="skyblue", edgecolor="black", bins=30
        )

        # 2. Curva KDE superpuesta (línea roja independiente)
        ax2 = ax.twinx()
        sns.kdeplot(
            df[var], ax=ax2, color="red", linewidth=2
        )
        ax2.grid(False)      # Desactiva la malla secundaria
        ax2.set_ylabel("")   # Elimina etiqueta del eje secundario
        ax2.set_yticks([])   # Oculta las marcas numéricas de la densidad

        # Formato del gráfico
        ax.set_title(f"Distribución e Histograma: {var}", fontsize=13, fontweight="bold")
        ax.set_xlabel(var, fontsize=11)
        ax.set_ylabel("Frecuencia (N° Registros)", fontsize=11)

        plt.tight_layout()

        plt.show()

def graficar_distribuciones_categoricas(df, columnas=None):
    """
    Genera gráficos de barras con frecuencias relativas (%) para variables categóricas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset con los datos de la planta.
    columnas : list, optional
        Lista de variables categóricas a graficar.
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
    
    plt.show()