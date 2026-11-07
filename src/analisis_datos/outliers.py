import pandas as pd
import numpy as np
from pathlib import Path





def verificar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspecciona y reporta la presencia de valores nulos o faltantes en el Dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset de entrada a auditar.
        
    Returns:
    --------
    pd.DataFrame
        Tabla con la frecuencia absoluta y relativa de nulos por variable.
    """
    total_filas = len(df)
    nulos_por_columna = df.isnull().sum()
    porcentaje_nulos = (nulos_por_columna / total_filas) * 100
    
    # Crear un DataFrame de resumen
    resumen_nulos = pd.DataFrame({
        'Variable': df.columns,
        'Valores_Nulos': nulos_por_columna.values,
        'Porcentaje (%)': porcentaje_nulos.values
    }).sort_values(by='Valores_Nulos', ascending=False).reset_index(drop=True)
    
    # Imprimir en pantalla el reporte formal
    print("=" * 60)
    print("      AUDITORÍA DE CALIDAD DE DATOS: VALORES VACÍOS")
    print("=" * 60)
    print(f"Total de registros evaluados: {total_filas:,} filas.")
    print(f"Total de variables evaluadas: {len(df.columns)} columnas.")
    print("-" * 60)
    
    total_nulos_dataset = nulos_por_columna.sum()
    
    if total_nulos_dataset == 0:
        print("RESULTADO: Integridad del 100%. No se detectaron valores nulos o vacios en ninguna variable.")
    else:
        print(f"ALERTA: Se detectaron un total de {total_nulos_dataset} valores nulos en el dataset.")
    
    print("-" * 60)
    print(resumen_nulos.to_string(index=False))
    print("=" * 60)
    
    return resumen_nulos

def generar_tabla_iqr(df, variables_num):
    """Calcula los percentiles Q1, Q3, IQR, limites y conteo de outliers

    para una lista de variables cuantitativas.
    """
    resumen_iqr = []

    for var in variables_num:
        q1 = df[var].quantile(0.25)
        median = df[var].median()
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr

        outliers_inf = df[df[var] < lim_inf].shape[0]
        outliers_sup = df[df[var] > lim_sup].shape[0]
        total_outliers = outliers_inf + outliers_sup
        porcentaje = (total_outliers / len(df)) * 100

        resumen_iqr.append({
            'Variable': var,
            'Q1 (25%)': round(q1, 2),
            'Mediana (50%)': round(median, 2),
            'Q3 (75%)': round(q3, 2),
            'IQR': round(iqr, 2),
            'Límite Inferior': round(lim_inf, 2),
            'Límite Superior': round(lim_sup, 2),
            'N° Outliers': total_outliers,
            '% Outliers': f'{porcentaje:.2f}%',
        })

    return pd.DataFrame(resumen_iqr)

def auditar_lagging_pf_critico(df, umbral_pf=40.0):
    """Filtra e inspecciona las lecturas de Lagging_Current_Power_Factor por debajo

    del umbral critico (0-40%) para analizar su relacion con Usage_kWh.
    """
    col_pf = "Lagging_Current_Power_Factor"
    col_usage = "Usage_kWh"
    col_reactiva = "Lagging_Current_Reactive_Power_kVarh"
    col_load = "Load_Type"

    # Filtrar lecturas criticas
    df_critico = df[df[col_pf] < umbral_pf].copy()
    total_criticos = len(df_critico)
    porcentaje = (total_criticos / len(df)) * 100

    print(
        f"=== 1. AUDITORÍA DE LAGGING POWER FACTOR CRÍTICO (< {umbral_pf}%) ==="
    )
    print(
        f"Registros encontrados: {total_criticos} de {len(df)} ({porcentaje:.4f}%)\n"
    )

    if total_criticos > 0:
        # Calcular estadisticos clave del consumo activo durante las fallas
        usage_min = df_critico[col_usage].min()
        usage_max = df_critico[col_usage].max()
        usage_mean = df_critico[col_usage].mean()
        usage_cero = (df_critico[col_usage] == 0).sum()

        print("Estadísticos del Consumo Activo (Usage_kWh) en las lecturas críticas:")
        print(f"  - Consumo Mínimo : {usage_min:.4f} kWh")
        print(f"  - Consumo Máximo : {usage_max:.4f} kWh")
        print(f"  - Consumo Promedio: {usage_mean:.4f} kWh")
        print(
            f"  - Registros con Consumo == 0: {usage_cero} de {total_criticos}"
        )

        print("\nDetalle de los registros críticos encontrados:")
        columnas_mostrar = [
            c
            for c in ["date", col_usage, col_reactiva, col_pf, col_load]
            if c in df.columns
        ]
        print(df_critico[columnas_mostrar].to_string(index=False))
    else:
        print("No se encontraron registros por debajo del umbral especificado.")

    print("\n" + "=" * 65 + "\n")
    return df_critico


def auditar_resolucion_co2(df, top_n=10):
    """Analiza la estructura discreta y el paso entre decimales de la variable CO2(tCO2)

    para fundamentar el efecto de cuantización (redondeo a 2 decimales).
    """
    col_co2 = "CO2(tCO2)"
    col_usage = "Usage_kWh"

    print("=== 2. AUDITORÍA DE RESOLUCIÓN Y REDONDEO EN CO2(tCO2) ===")

    # 1. Obtener valores unicos y pasos/diferencias numéricas
    valores_unicos = np.sort(df[col_co2].unique())
    diferencias = np.round(np.diff(valores_unicos), 4)
    paso_minimo = (
        np.min(diferencias[diferencias > 0]) if len(diferencias) > 0 else 0
    )

    print(
        f"Total de valores únicos que toma CO2(tCO2): {len(valores_unicos)}"
    )
    print(f"Paso mínimo discreto detectado entre valores: {paso_minimo:.4f} tCO2")
    print(f"Primeros 10 valores únicos en orden ascendente: {valores_unicos[:10]}")

    # 2. Conteo de frecuencias de los valores más comunes
    print(
        f"\nTop {top_n} valores más frecuentes de CO2(tCO2) (Efecto Muestra Discreta):"
    )
    frecuencias = df[col_co2].value_counts().head(top_n).reset_index()
    frecuencias.columns = ["Valor_CO2_tCO2", "Frecuencia_Absoluta"]
    frecuencias["Proporcion_%"] = np.round(
        (frecuencias["Frecuencia_Absoluta"] / len(df)) * 100, 2
    )
    print(frecuencias.to_string(index=False))

    # 3. Discrepancia del tramo base (CO2 <= 0.01 vs Usage_kWh <= 10)
    co2_base = (df[col_co2] <= 0.01).mean() * 100
    usage_base = (df[col_usage] <= 10).mean() * 100

    print(
        "\nComparativa de tramos de reposo (Pérdida de sensibilidad por redondeo):"
    )
    print(f"  - Proporción CO2 en tramo bajo (0 - 0.01 tCO2): {co2_base:.2f}%")
    print(
        f"  - Proporción Usage_kWh en tramo reposo (0 - 10 kWh): {usage_base:.2f}%"
    )
    print(
        f"  - Desfase absoluto por truncamiento de decimales: {abs(co2_base - usage_base):.2f}%"
    )
    print("=" * 65)

def imputar_lagging_pf_incoherente(df):
    """Detecta e imputa el único registro incoherente de Lagging_Current_Power_Factor

    (07/11/2018 00:00 h con 0.00% y consumo 0.00 kWh) mediante interpolación
    lineal.
    """
    col_pf = "Lagging_Current_Power_Factor"
    col_usage = "Usage_kWh"

    df_procesado = df.copy()
    total_inicial = len(df_procesado)

    mask_incoherente = (df_procesado[col_pf] == 0) & (
        df_procesado[col_usage] == 0
    )
    total_imputados = mask_incoherente.sum()

    if total_imputados > 0:
        df_procesado.loc[mask_incoherente, col_pf] = np.nan
        df_procesado[col_pf] = df_procesado[col_pf].interpolate(method="linear")

    registros_eliminados = 0
    registros_retenidos = len(df_procesado)
    pct_imputados = (total_imputados / total_inicial) * 100

    print("=== DIAGNÓSTICO DE IMPACTO MUESTRAL Y TRATAMIENTO DE OUTLIERS ===")
    print(f"  - Total de Registros Iniciales : {total_inicial:,} filas (100.00%)")
    print(
        "  - Registros Eliminados          :"
        f" {registros_eliminados} filas (0.00%)"
    )
    print(
        f"  - Registros Retenidos          : {registros_retenidos:,} filas"
        " (100.00%)"
    )
    print(
        "  - Registros Imputados/Modificados:"
        f" {total_imputados} registro ({pct_imputados:.3f}%)"
    )
    print("=" * 65 + "\n")

    return df_procesado


def exportar_datos_procesados(
    df, ruta_salida="../data/clean/Steel_industry_clean.csv"
):
    """Guarda el DataFrame procesado en la ruta especificada."""
    ruta = Path(ruta_salida).resolve()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"[✓] Dataset limpio exportado exitosamente a: {ruta}")