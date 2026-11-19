"""Cross validation de los SVM y búsqueda de hiperparámetros - Tarea 5.

Idea de la validación cruzada
-----------------------------
Un solo corte entrenamiento/prueba puede dar un resultado favorable o desfavorable
por azar. En la validación cruzada de k folds los datos se dividen en k partes; cada
parte hace de "prueba" una vez mientras el modelo se entrena con las otras k-1. Se
reporta el promedio y la desviación estándar de cada métrica: el promedio dice qué
tan bien funciona el SVM y la desviación qué tan estable es.

Dos esquemas de partición
-------------------------
- "estratificado": k folds al azar que conservan la proporción de alertas en cada
  fold (el esquema clásico).
- "temporal": los folds respetan el orden del tiempo (se entrena con el pasado y se
  valida con el futuro, dejando un hueco de `gap` filas entre ambos). Es más exigente
  y más realista, porque las lecturas son cada 15 min y las vecinas se parecen mucho:
  mezclarlas al azar puede dar resultados optimistas.

La validación cruzada se hace SOLO con el conjunto de entrenamiento; el conjunto de
prueba (octubre-diciembre) queda intacto para la evaluación final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     TimeSeriesSplit, cross_validate)

from . import datos
from . import modelos_svm as ms

SCORING = {
    "Exactitud": "accuracy",
    "Exactitud_balanceada": "balanced_accuracy",
    "Recall": "recall",
    "Precision": "precision",
    "F1": "f1",
    "AUC": "roc_auc",
    "AP": "average_precision",
}
METRICAS_CLAVE = ["Recall", "Precision", "F1", "AUC"]

# Grillas pequeñas de hiperparámetros (nombres sin el prefijo del pipeline).
GRILLAS = {
    "lineal":     {"C": [0.01, 0.1, 1, 10]},
    "polinomial": {"C": [0.1, 1, 10], "degree": [2, 3]},
    "rbf":        {"C": [0.1, 1, 10], "gamma": [0.01, 0.05, "scale", 0.3]},
}


def crear_esquema(esquema: str = "estratificado", k: int = 5, gap: int = 96,
                  semilla: int = ms.SEMILLA):
    """Devuelve el objeto de partición de scikit-learn.

    esquema='estratificado' -> StratifiedKFold(k, shuffle=True)
    esquema='temporal'      -> TimeSeriesSplit(k, gap=gap)  (gap=96 filas = 1 día)
    Para 'temporal' los datos deben estar ordenados cronológicamente (lo están).
    """
    if esquema == "estratificado":
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
    if esquema == "temporal":
        return TimeSeriesSplit(n_splits=k, gap=gap)
    raise ValueError("esquema debe ser 'estratificado' o 'temporal'")


def validar_kernel(nombre_kernel: str, X, y, esquema: str = "estratificado", k: int = 5,
                   class_weight="balanced", **params) -> pd.DataFrame:
    """Cross validation de un SVM. Devuelve una fila por fold con todas las métricas."""
    modelo = ms.crear_svm(nombre_kernel, class_weight=class_weight, **params)
    res = cross_validate(modelo, X, y, cv=crear_esquema(esquema, k), scoring=SCORING,
                         n_jobs=1, error_score="raise")
    filas = pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})
    filas["Seg_entrenamiento"] = res["fit_time"]
    filas.index = [f"fold_{i + 1}" for i in range(len(filas))]
    return filas


def comparar_kernels_cv(X, y, esquema: str = "estratificado", k: int = 5,
                        class_weight="balanced", parametros: dict | None = None):
    """Cross validation de los tres kernels con el mismo esquema de folds.

    Devuelve (medias, desviaciones, folds): dos tablas (kernel x métrica) y un dict
    con el detalle por fold de cada kernel.
    """
    parametros = parametros or {}
    folds = {n: validar_kernel(n, X, y, esquema, k, class_weight, **parametros.get(n, {}))
             for n in ms.KERNELS}
    medias = pd.DataFrame({n: f.mean() for n, f in folds.items()}).T
    desv = pd.DataFrame({n: f.std() for n, f in folds.items()}).T
    medias.index.name = desv.index.name = "Kernel"
    return medias, desv, folds


def linea_base_cv(X, y, esquema: str = "estratificado", k: int = 5):
    """Mide la regla de persistencia sobre los mismos folds (referencia a superar)."""
    filas = []
    for _, idx_val in crear_esquema(esquema, k).split(X, y):
        pred = datos.linea_base_persistencia(X.iloc[idx_val], y.iloc[idx_val])
        filas.append(ms.metricas_clasificacion(y.iloc[idx_val], pred))
    f = pd.DataFrame(filas)
    return f.mean(), f.std()


def formatear_media_desv(medias: pd.DataFrame, desv: pd.DataFrame,
                         metricas=METRICAS_CLAVE, decimales: int = 3) -> pd.DataFrame:
    """Tabla de texto 'media ± desviación' lista para mostrar en el informe."""
    out = pd.DataFrame(index=medias.index)
    for m in metricas:
        out[m] = [f"{a:.{decimales}f} ± {b:.{decimales}f}"
                  for a, b in zip(medias[m], desv[m])]
    if "Seg_entrenamiento" in medias:
        out["Seg_por_fold"] = medias["Seg_entrenamiento"].round(2)
    return out


def mejor_kernel(medias: pd.DataFrame, criterio: str = "F1") -> str:
    """Nombre del kernel con mayor promedio en la métrica `criterio`."""
    return str(medias[criterio].idxmax())


def ajustar_hiperparametros(nombre_kernel: str, X, y, grilla: dict | None = None,
                            esquema: str = "estratificado", k: int = 5,
                            scoring: str = "f1", class_weight="balanced"):
    """Búsqueda en grilla (GridSearchCV) de los hiperparámetros de un kernel.

    Usa la misma validación cruzada para elegir la mejor combinación según `scoring`.
    Devuelve (mejores_parametros, tabla_resultados). Ojo: el mejor puntaje de la
    búsqueda es algo optimista (se eligió mirando esos mismos folds); la estimación
    honesta es la evaluación final en el conjunto de prueba.
    """
    grilla = grilla or GRILLAS[nombre_kernel]
    modelo = ms.crear_svm(nombre_kernel, class_weight=class_weight)
    busqueda = GridSearchCV(modelo, {f"svm__{p}": v for p, v in grilla.items()},
                            cv=crear_esquema(esquema, k), scoring=scoring,
                            n_jobs=1, refit=False)
    busqueda.fit(X, y)
    r = pd.DataFrame(busqueda.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score"]
    tabla = r[cols].rename(columns=lambda c: c.replace("param_svm__", "")).sort_values(
        "rank_test_score").reset_index(drop=True)
    mejores = {p.replace("svm__", ""): v for p, v in busqueda.best_params_.items()}
    return mejores, tabla


def ajustar_todos(X, y, esquema: str = "estratificado", k: int = 5, scoring: str = "f1",
                  class_weight="balanced", grillas: dict | None = None):
    """Ajusta los tres kernels. Devuelve (dict de mejores parámetros, dict de tablas)."""
    grillas = grillas or GRILLAS
    mejores, tablas = {}, {}
    for n in ms.KERNELS:
        mejores[n], tablas[n] = ajustar_hiperparametros(
            n, X, y, grillas[n], esquema, k, scoring, class_weight)
    return mejores, tablas
