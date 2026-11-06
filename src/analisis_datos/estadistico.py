"""
Módulo para el procesamiento y presentación de estadísticos descriptivos.
Proyecto Daewoo Steel Co.
"""

import pandas as pd


def calcular_estadisticos_cuantitativos(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Calcula las medidas de tendencia central, dispersión y forma para las variables numéricas."""
    resultados = []

    for col in variables:
        if col not in df.columns:
            continue

        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25

        resultados.append({
            'Variable': col,
            'Media': round(df[col].mean(), 2),
            'Mediana': round(df[col].median(), 2),
            'Desv. Est.': round(df[col].std(), 2),
            'Mínimo': round(df[col].min(), 2),
            'Máximo': round(df[col].max(), 2),
            'IQR': round(iqr, 2),
            'Asimetría (Skew)': round(df[col].skew(), 2),
            'Curtosis (Kurt)': round(df[col].kurt(), 2),
        })

    return pd.DataFrame(resultados)


def imprimir_resumen_cuantitativo(
    df_resumen: pd.DataFrame, total_registros: int = 35040
) -> None:
    """Da formato e imprime en consola el resumen cuantitativo."""
    print("\n" + "=" * 80)
    print(
        f"SECCIÓN 1.3: RESUMEN CUANTITATIVO DE VARIABLES NUMÉRICAS (N ="
        f" {total_registros:,})"
    )
    print("=" * 80)
    print(df_resumen.to_string(index=False))
    print("=" * 80 + "\n")


def calcular_rangos_variable(df, columna, bins, labels=None):
    """
    Agrupa cualquier variable continua en tramos/rangos específicos 
    y retorna la tabla de frecuencias absolutas y relativas (%).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset con los datos de la planta.
    columna : str
        Nombre de la variable continua a categorizar.
    bins : list
        Límites de los intervalos (ej. [0, 10, 50, 100, 160]).
    labels : list, optional
        Nombres descriptivos para cada rango.
    """
    df_rangos = pd.cut(df[columna], bins=bins, labels=labels, include_lowest=True)
    frec_abs = df_rangos.value_counts().reindex(labels if labels else None)
    frec_rel = (frec_abs / len(df) * 100).round(2)
    
    tabla = pd.DataFrame({
        "Variable": columna,
        "Frecuencia Absoluta (N°)": frec_abs,
        "Proporción Relativa (%)": frec_rel.map("{:.2f}%".format)
    })
    tabla.index.name = f"Rango / Tramo ({columna})"
    return tabla

import pandas as pd

def analizar_rangos_variables_continuas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa las 7 variables cuantitativas clave en tramos y rangos operativos,
    retornando una única tabla consolidada de frecuencias y proporciones.
    """
    configuraciones = [
        {
            "columna": "Usage_kWh",
            "bins": [-0.1, 10, 50, 100, 160],
            "labels": ["Reposo/Base (0-10 kWh)", "Carga Media (10-50 kWh)", "Carga Alta (50-100 kWh)", "Picos Extremos (>100 kWh)"]
        },
        {
            "columna": "Lagging_Current_Reactive.Power_kVarh",
            "bins": [-0.1, 5, 20, 50, 100],
            "labels": ["Reposo Inductivo (0-5 kVarh)", "Carga Inductiva Media (5-20 kVarh)", "Carga Inductiva Alta (20-50 kVarh)", "Picos Inductivos Extremos (>50 kVarh)"]
        },
        {
            "columna": "Leading_Current_Reactive_Power_kVarh",
            "bins": [-0.1, 1, 10, 30],
            "labels": ["Sin Inyección Capacitiva (0-1 kVarh)", "Inyección Capacitiva Baja (1-10 kVarh)", "Inyección Capacitiva Alta (>10 kVarh)"]
        },
        {
            "columna": "CO2(tCO2)",
            "bins": [-0.001, 0.01, 0.03, 0.05, 0.08],
            "labels": ["Baja Emisión (0-0.01)", "Emisión Moderada (0.01-0.03)", "Emisión Alta (0.03-0.05)", "Emisión Crítica (>0.05)"]
        },
        {
            "columna": "Lagging_Current_Power_Factor",
            "bins": [-0.1, 40, 70, 90, 100],
            "labels": ["Falla/Crítico (0-40%)", "Baja Eficiencia (40-70%)", "Aceptable (70-90%)", "Óptimo (90-100%)"]
        },
        {
            "columna": "Leading_Current_Power_Factor",
            "bins": [-0.1, 50, 99.9, 100],
            "labels": ["Baja Regulación (0-50%)", "Regulación Media (50-99.9%)", "Compensación Total (100%)"]
        },
        {
            "columna": "NSM",
            "bins": [-0.1, 21600, 43200, 64800, 85500],
            "labels": ["Madrugada (00:00-06:00)", "Mañana (06:00-12:00)", "Tarde (12:00-18:00)", "Noche (18:00-24:00)"]
        }
    ]
    
    lista_tablas = []
    
    for cfg in configuraciones:
        col = cfg["columna"]
        if col in df.columns:
            df_rangos = pd.cut(df[col], bins=cfg["bins"], labels=cfg["labels"], include_lowest=True)
            frec_abs = df_rangos.value_counts().reindex(cfg["labels"])
            frec_rel = (frec_abs / len(df) * 100).round(2)
            
            df_temp = pd.DataFrame({
                "Variable": col,
                "Rango / Tramo Operativo": frec_abs.index,
                "Frecuencia Absoluta (N°)": frec_abs.values,
                "Proporción Relativa (%)": frec_rel.map("{:.2f}%".format).values
            })
            lista_tablas.append(df_temp)
            
    return pd.concat(lista_tablas, ignore_index=True)