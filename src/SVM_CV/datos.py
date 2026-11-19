"""Carga de datos y construcción del problema de alerta (Tarea 5 - SVM).

Problema
--------
Con lo que se sabe en el instante t (consumo reciente, carga actual, mediciones
eléctricas y calendario del instante pronosticado), predecir si la planta estará
en **carga máxima** (`Maximum_Load`) dentro de `horizonte` intervalos de 15 min.

    clase 1 (alerta)  : Maximum_Load en t + horizonte
    clase 0 (normal)  : Light_Load o Medium_Load en t + horizonte

Las variables y la partición cronológica (entrenamiento < 2018-10-01 <= prueba)
son las mismas de la Tarea 4, así que se reutiliza `modelos_aprendizaje.pronostico`
para que ambos trabajos sean comparables.

Uso típico (desde el notebook)
------------------------------
    from SVM_CV import datos
    df = datos.cargar_dataset_limpio()
    X_tr, X_te, y_tr, y_te, info = datos.construir_problema(df, horizonte=4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modelos_aprendizaje import pronostico as pr

CLASE_ALERTA = "Maximum_Load"
NOMBRES_CLASES = {0: "Normal (Light/Medium)", 1: "Alerta (Maximum_Load)"}


def encontrar_raiz() -> Path:
    """Raíz del proyecto, a partir de la ubicación de este archivo (src/SVM_CV/datos.py).

    No depende de la carpeta desde la que se ejecute el notebook.
    """
    return Path(__file__).resolve().parents[2]


def cargar_dataset_limpio(ruta: str | Path | None = None) -> pd.DataFrame:
    """Carga data/clean/Steel_industry_clean.csv (salida de la Tarea 2), ordenado por fecha."""
    ruta_csv = (
        Path(ruta)
        if ruta is not None
        else encontrar_raiz() / "data" / "clean" / "Steel_industry_clean.csv"
    )
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV limpio en: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df.sort_values("date").reset_index(drop=True)


def construir_problema(df: pd.DataFrame, horizonte: int = 4,
                       clase_alerta: str = CLASE_ALERTA, fecha_corte=pr.FECHA_CORTE):
    """Arma X, y (binaria) y parte en entrenamiento y prueba por fecha pronosticada.

    Devuelve (X_tr, X_te, y_tr, y_te, info). `y` vale 1 si en t + horizonte la carga
    es `clase_alerta` y 0 en caso contrario. El índice de X es el instante de origen t.
    """
    X_tr, X_te, y_tr, y_te, info = pr.preparar_pronostico(
        df, horizonte=horizonte, fecha_corte=fecha_corte)
    y_tr = (y_tr == clase_alerta).astype(int)
    y_te = (y_te == clase_alerta).astype(int)
    info = dict(info)
    info["clase_alerta"] = clase_alerta
    info["horizonte_minutos"] = horizonte * pr.PASO_MIN
    return X_tr, X_te, y_tr, y_te, info


def resumen_clases(y_tr, y_te) -> pd.DataFrame:
    """Cuántos casos hay de cada clase en entrenamiento y prueba (para ver el desbalance)."""
    filas = []
    for nombre, y in (("Entrenamiento", y_tr), ("Prueba", y_te)):
        n1 = int((y == 1).sum())
        filas.append({
            "Conjunto": nombre,
            "Casos": len(y),
            "Normal (0)": len(y) - n1,
            "Alerta (1)": n1,
            "% Alerta": 100 * n1 / len(y),
        })
    return pd.DataFrame(filas)


def submuestra_estratificada(X, y, n: int | None, semilla: int = 42):
    """Submuestra de `n` filas manteniendo la proporción de clases y el orden cronológico.

    El SVM con kernel no lineal tiene un costo que crece muy rápido con el número de
    filas (entre n² y n³), por eso la validación cruzada se hace con una muestra. Se
    conserva el orden temporal para poder usar también la validación por bloques.
    Con n=None (o n >= len(X)) devuelve todo sin cambios.
    """
    if n is None or n >= len(X):
        return X, y
    rng = np.random.default_rng(semilla)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    n_pos = int(round(n * len(pos) / len(y)))
    n_neg = n - n_pos
    elegidos = np.sort(np.concatenate([
        rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
        rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
    ]))
    return X.iloc[elegidos], y.iloc[elegidos]


def linea_base_persistencia(X, y) -> dict:
    """Regla trivial: 'estará en carga máxima en 1 h si hoy ya está en carga máxima'.

    Es la referencia a superar: si el SVM no mejora esto, el modelo no aporta.
    Devuelve las predicciones (0/1) para evaluarlas con las mismas métricas.
    """
    pred = (X["Carga_actual"].astype(int) == pr.ORDEN_CARGA.index(CLASE_ALERTA)).astype(int)
    return pred.to_numpy()
