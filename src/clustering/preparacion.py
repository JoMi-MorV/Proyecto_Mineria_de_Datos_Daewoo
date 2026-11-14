"""Carga y preparación de datos para KNN (clasificación y regresión)."""
from pathlib import Path

import numpy as np
import pandas as pd

USAGE = "Usage_kWh"
LAG_KVARH = "Lagging_Current_Reactive.Power_kVarh"
LEAD_KVARH = "Leading_Current_Reactive_Power_kVarh"
LAG_PF = "Lagging_Current_Power_Factor"
LEAD_PF = "Leading_Current_Power_Factor"
CO2 = "CO2(tCO2)"
ORDEN_CARGA = ["Light_Load", "Medium_Load", "Maximum_Load"]

# Variables eléctricas "técnicas". CO2 y Usage_kWh NO entran como predictoras
# de Load_Type, y CO2 tampoco entra al predecir Usage_kWh (fuga de información).
FEATURES_ELECTRICAS = [LAG_KVARH, LEAD_KVARH, LAG_PF, LEAD_PF]


def encontrar_raiz(inicio=None) -> Path:
    """Sube por las carpetas hasta encontrar la que contiene data/ y src/."""
    p = Path(inicio or Path.cwd()).resolve()
    for c in [p, *p.parents]:
        if (c / "data").is_dir() and (c / "src").is_dir():
            return c
    raise FileNotFoundError("No se encontró la raíz del proyecto (data/ y src/).")


def cargar_dataset_limpio(ruta=None) -> pd.DataFrame:
    """Carga data/clean/Steel_industry_clean.csv (salida de la Tarea 2)."""
    ruta = Path(ruta) if ruta else encontrar_raiz() / "data" / "clean" / "Steel_industry_clean.csv"
    df = pd.read_csv(ruta)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df.sort_values("date").reset_index(drop=True)


def _variables_base(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["WeekStatus_bin"] = (d["WeekStatus"] == "Weekend").astype(int)
    ang = 2 * np.pi * d["NSM"] / 86400
    d["NSM_sin"], d["NSM_cos"] = np.sin(ang), np.cos(ang)
    return d


def preparar_clasificacion(df: pd.DataFrame, nsm_ciclico: bool = True):
    """X, y, fechas para predecir Load_Type (sin Usage_kWh ni CO2).

    nsm_ciclico=True reemplaza NSM por sin/cos, así 23:45 y 00:00 quedan
    cerca en el espacio de distancias (importante para KNN).
    """
    d = _variables_base(df)
    tiempo = ["NSM_sin", "NSM_cos"] if nsm_ciclico else ["NSM"]
    feats = FEATURES_ELECTRICAS + ["WeekStatus_bin"] + tiempo
    return d[feats], d["Load_Type"], d["date"]


def preparar_regresion(df: pd.DataFrame, horizonte: int = 0,
                       rezagos=(1, 2, 4, 96), nsm_ciclico: bool = True):
    """X, y, fechas para predecir Usage_kWh (sin CO2).

    horizonte = 0  -> estimación del consumo del mismo instante (nowcasting).
    horizonte = h  -> pronóstico h pasos de 15 min adelante (16 = 4 h, 96 = 24 h).
                      Usa el estado actual, rezagos del consumo y el calendario
                      del instante objetivo (que sí se conoce de antemano).
    Devuelve 'fechas' = fecha del instante objetivo (sirve para el corte cronológico).
    """
    d = _variables_base(df)
    tiempo = ["NSM_sin", "NSM_cos"] if nsm_ciclico else ["NSM"]
    tiempo += ["WeekStatus_bin"]
    feats = list(FEATURES_ELECTRICAS)

    if horizonte > 0:
        d["Usage_actual"] = d[USAGE]
        feats.append("Usage_actual")
        for r in rezagos:
            d[f"Usage_t-{r}"] = d[USAGE].shift(r)
            feats.append(f"Usage_t-{r}")
        for c in tiempo:                      # calendario del instante objetivo
            d[c + "_obj"] = d[c].shift(-horizonte)
        feats += [c + "_obj" for c in tiempo]
    else:
        feats += tiempo

    d["y"] = d[USAGE].shift(-horizonte)
    d["fecha_obj"] = d["date"].shift(-horizonte)
    d = d.dropna(subset=feats + ["y", "fecha_obj"])
    return d[feats], d["y"], d["fecha_obj"]


def split_cronologico(X, y, fechas, corte: str = "2018-10-01"):
    """Entrena con fechas < corte y prueba con fechas >= corte (sin mezclar el tiempo)."""
    m = (fechas < pd.Timestamp(corte)).to_numpy()
    return X[m], X[~m], y[m], y[~m]
