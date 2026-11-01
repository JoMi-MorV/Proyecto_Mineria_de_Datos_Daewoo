"""
Módulo de ingesta y gestión de datos.
Proyecto Daewoo Steel Co.
"""

from pathlib import Path
import pandas as pd


def obtener_ruta_raiz() -> Path:
    """Devuelve la ruta absoluta a la raíz del proyecto."""
    # Asume que este archivo está en src/analisis_datos/ingesta.py
    return Path(__file__).resolve().parents[2]


def cargar_datos_raw(nombre_archivo: str = "Steel_industry_data.csv") -> pd.DataFrame:
    """Carga el conjunto de datos original desde la carpeta data/raw/.

    Args:
        nombre_archivo (str): Nombre del archivo CSV en la carpeta data/raw.

    Returns:
        pd.DataFrame: DataFrame cargado con las observaciones del dataset.
    """
    ruta_raiz = obtener_ruta_raiz()
    ruta_csv = ruta_raiz / "data" / "raw" / nombre_archivo

    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {ruta_csv.resolve()}"
        )

    print(f"[✓] Cargas exitosa desde: data/raw/{nombre_archivo}")
    return pd.read_csv(ruta_csv)