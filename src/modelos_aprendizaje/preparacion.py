"""Carga del dataset limpio para los modelos de la Tarea 4."""
from pathlib import Path

import pandas as pd


def cargar_dataset_limpio(ruta: str | Path | None = None) -> pd.DataFrame:
    """Carga y ordena por fecha el CSV limpio generado en la Tarea 2."""
    ruta_csv = (
        Path(ruta)
        if ruta is not None
        else Path(__file__).resolve().parents[2]
        / "data"
        / "clean"
        / "Steel_industry_clean.csv"
    )
    df = pd.read_csv(ruta_csv)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df.sort_values("date").reset_index(drop=True)
