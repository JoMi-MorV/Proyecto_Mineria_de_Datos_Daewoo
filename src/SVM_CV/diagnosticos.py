"""Diagnósticos opcionales de la Tarea 5 (anexos A y B).

Comprueban con experimentos dos explicaciones que el cuerpo principal solo describe:

Anexo A - ¿por qué el primer fold de la validación temporal rinde peor?
    - `primer_fold_vs_azar`: mismo número de casos, distinta cobertura en el tiempo.
    - `curva_aprendizaje`: cómo cambia el F1 de cada kernel al aumentar los datos.

Anexo B - ¿el RBF con `gamma` alto memoriza el entrenamiento?
    - `curva_validacion_gamma`: puntaje de entrenamiento vs validación (CV temporal,
      sin usar el conjunto de prueba).
    - `diagnostico_gamma_en_prueba`: el mismo barrido evaluado en prueba. Es solo un
      diagnóstico: no debe usarse para elegir hiperparámetros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, validation_curve

from . import datos
from . import modelos_svm as ms
from . import validacion_cruzada as vc

KERNELES = list(ms.KERNELS)


# ---------------------------------------------------------------------------
# Anexo A
# ---------------------------------------------------------------------------
def primer_fold_vs_azar(X, y, fechas_obj, k: int = 5, gap: int = 96, semillas: int = 5):
    """Compara el primer fold de la CV temporal con un entrenamiento del mismo tamaño
    tomado al azar del resto del año.

    Ambos se validan en el mismo bloque (el del fold 1). Si el rendimiento sube cuando
    los casos vienen de todo el año, la debilidad del fold 1 no se debe a la cantidad
    de datos sino a que sus primeras semanas no representan el periodo siguiente.

    Devuelve (tabla, descripcion). `fechas_obj` es la fecha pronosticada de cada fila
    (info["fecha_objetivo_tr"]).
    """
    idx_tr, idx_va = next(iter(TimeSeriesSplit(n_splits=k, gap=gap).split(X)))
    Xv, yv = X.iloc[idx_va], y.iloc[idx_va]
    excluidos = np.concatenate([idx_tr, np.arange(idx_va[0] - gap, idx_va[-1] + gap + 1)])
    resto = np.setdiff1d(np.arange(len(X)), excluidos)

    filas = []
    for nombre in KERNELES:
        m, _ = ms.entrenar_svm(nombre, X.iloc[idx_tr], y.iloc[idx_tr])
        met = ms.metricas_clasificacion(yv, m.predict(Xv), m.decision_function(Xv))
        filas.append({"Entrenamiento": "Primeras semanas (fold real)", "Kernel": nombre, **met})

        mets = []
        for s in range(semillas):
            idx = np.random.default_rng(s).choice(resto, size=len(idx_tr), replace=False)
            m, _ = ms.entrenar_svm(nombre, X.iloc[idx], y.iloc[idx])
            mets.append(ms.metricas_clasificacion(yv, m.predict(Xv), m.decision_function(Xv)))
        filas.append({"Entrenamiento": "Mismo n, al azar del resto del año", "Kernel": nombre,
                      **pd.DataFrame(mets).mean().to_dict()})
    desc = {
        "n_entrenamiento": len(idx_tr),
        "n_validacion": len(idx_va),
        "entrena_desde": fechas_obj.iloc[idx_tr[0]], "entrena_hasta": fechas_obj.iloc[idx_tr[-1]],
        "valida_desde": fechas_obj.iloc[idx_va[0]], "valida_hasta": fechas_obj.iloc[idx_va[-1]],
    }
    tabla = pd.DataFrame(filas)[["Entrenamiento", "Kernel", "Recall", "Precision", "F1", "AUC"]]
    return tabla, desc


def curva_aprendizaje(X, y, fechas_obj, corte: str = "2018-07-01",
                      tamanos=(300, 600, 1200, 2400, 4800, 9600, None), semillas: int = 5):
    """F1 y AUC de cada kernel al entrenar con n casos y validar siempre en el mismo bloque.

    Se entrena con casos al azar de los datos anteriores a `corte` (tamaño None = todos)
    y se valida con los posteriores a `corte` (dentro del entrenamiento; la prueba no se toca).
    """
    es_antes = (fechas_obj < pd.Timestamp(corte)).to_numpy()
    Xa, ya = X[es_antes], y[es_antes]
    Xv, yv = X[~es_antes], y[~es_antes]
    filas = []
    for n in tamanos:
        n_real = len(Xa) if n is None else n
        reps = 1 if n is None else semillas
        for s in range(reps):
            idx = (np.arange(len(Xa)) if n is None else
                   np.sort(np.random.default_rng(s).choice(len(Xa), size=n, replace=False)))
            for nombre in KERNELES:
                m, _ = ms.entrenar_svm(nombre, Xa.iloc[idx], ya.iloc[idx])
                met = ms.metricas_clasificacion(yv, m.predict(Xv), m.decision_function(Xv))
                filas.append({"n": n_real, "semilla": s, "Kernel": nombre,
                              "F1": met["F1"], "AUC": met["AUC"]})
    r = pd.DataFrame(filas)
    r.attrs["n_base"] = len(Xa)
    r.attrs["n_validacion"] = len(Xv)
    r.attrs["alerta_validacion"] = float(yv.mean())
    return r


def resumen_curva_aprendizaje(r: pd.DataFrame, metrica: str = "F1") -> pd.DataFrame:
    """Promedio de la métrica por tamaño (filas) y kernel (columnas)."""
    return r.groupby(["n", "Kernel"])[metrica].mean().unstack("Kernel")[KERNELES]


# ---------------------------------------------------------------------------
# Anexo B
# ---------------------------------------------------------------------------
def curva_validacion_gamma(X, y, gammas=(0.005, 0.01, 0.05, 0.1, 0.3, 1.0), C: float = 10,
                           n_muestra: int | None = 8000, k: int = 5):
    """Puntaje de entrenamiento y de validación (CV temporal) del RBF para cada `gamma`.

    Una brecha grande entre ambos indica que el modelo memoriza el entrenamiento.
    Devuelve una tabla en formato largo con las métricas AUC y F1.
    """
    Xs, ys = datos.submuestra_estratificada(X, y, n_muestra)
    modelo = ms.crear_svm("rbf", C=C)
    filas = []
    for nombre_score, score in (("AUC", "roc_auc"), ("F1", "f1")):
        tr, va = validation_curve(modelo, Xs, ys, param_name="svm__gamma",
                                  param_range=list(gammas),
                                  cv=vc.crear_esquema("temporal", k), scoring=score)
        for g, a, b in zip(gammas, tr.mean(1), va.mean(1)):
            filas.append({"gamma": g, "Metrica": nombre_score, "Entrenamiento": a,
                          "Validacion": b, "Brecha": a - b})
    return pd.DataFrame(filas)


def diagnostico_gamma_en_prueba(X_tr, X_te, y_tr, y_te,
                                gammas=(0.01, 0.05, "scale", 0.3, 1.0), C: float = 10):
    """RBF con cada `gamma`, entrenado con todo el entrenamiento y medido en prueba.

    SOLO DIAGNÓSTICO: la prueba no debe usarse para elegir `gamma`.
    """
    filas = []
    for g in gammas:
        m, _ = ms.entrenar_svm("rbf", X_tr, y_tr, C=C, gamma=g)
        tr = ms.metricas_clasificacion(y_tr, m.predict(X_tr), m.decision_function(X_tr))
        te = ms.evaluar_en_prueba(m, X_te, y_te)
        filas.append({"gamma": g, "AUC_entrenamiento": tr["AUC"], "AUC_prueba": te["AUC"],
                      "F1_entrenamiento": tr["F1"], "F1_prueba": te["F1"],
                      "Recall_prueba": te["Recall"], "Precision_prueba": te["Precision"]})
    return pd.DataFrame(filas).set_index("gamma")
