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