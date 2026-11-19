"""SVM con tres kernels (lineal, polinomial y RBF) - Tarea 5.

Cada SVM busca el hiperplano que separa 'alerta' de 'normal' con el mayor margen:
- lineal      : el hiperplano es un plano recto en el espacio de las variables.
- polinomial  : el hiperplano es recto en un espacio ampliado con productos entre
                variables, por lo que en el espacio original la frontera es curva.
- RBF         : la frontera se adapta localmente alrededor de los datos (muy flexible).

El SVM es sensible a la escala de las variables, por eso todo modelo es un `Pipeline`
con `StandardScaler` + `SVC`. El escalador se ajusta solo con los datos de
entrenamiento (de cada fold, durante la validación cruzada), sin filtrar información.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEMILLA = 42

# Hiperparámetros por defecto de cada kernel (valores razonables, sin ajustar).
KERNELS = {
    "lineal":     {"kernel": "linear", "C": 1.0},
    "polinomial": {"kernel": "poly", "C": 1.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
    "rbf":        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
}

DESCRIPCION_KERNELS = {
    "lineal": "Hiperplano recto en el espacio original.",
    "polinomial": "Frontera curva (grado 3): captura interacciones entre variables.",
    "rbf": "Frontera flexible y local: se adapta a la forma de los datos.",
}


def crear_svm(nombre_kernel: str, class_weight="balanced", **params) -> Pipeline:
    """Pipeline StandardScaler + SVC para el kernel pedido ('lineal', 'polinomial' o 'rbf').

    class_weight='balanced' pondera más la clase minoritaria (alerta), de modo que el
    modelo no aprenda a decir siempre 'normal'. Con None se usa el peso normal.
    `params` permite sobrescribir C, gamma, degree, etc.
    """
    if nombre_kernel not in KERNELS:
        raise ValueError(f"Kernel desconocido: {nombre_kernel}. Usa {list(KERNELS)}")
    config = {**KERNELS[nombre_kernel], **params}
    svc = SVC(class_weight=class_weight, cache_size=1000, random_state=SEMILLA, **config)
    return Pipeline([("escala", StandardScaler()), ("svm", svc)])


def entrenar_svm(nombre_kernel: str, X_tr, y_tr, class_weight="balanced", **params):
    """Entrena el SVM y devuelve (modelo, segundos de entrenamiento)."""
    modelo = crear_svm(nombre_kernel, class_weight=class_weight, **params)
    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    return modelo, time.perf_counter() - t0


def metricas_clasificacion(y_real, y_pred, puntaje=None) -> dict:
    """Métricas centradas en la clase 'alerta' (1).

    Recall    : de las cargas máximas reales, cuántas se anticiparon.
    Precisión : de las alertas emitidas, cuántas eran reales (1 - falsas alarmas).
    F1        : equilibrio entre recall y precisión.
    AUC / AP  : calidad del puntaje del modelo (distancia al hiperplano) sin fijar
                un umbral; requieren `puntaje` = decision_function.
    """
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    m = {
        "Exactitud": float((y_real == y_pred).mean()),
        "Exactitud_balanceada": balanced_accuracy_score(y_real, y_pred),
        "Recall": recall_score(y_real, y_pred, zero_division=0),
        "Precision": precision_score(y_real, y_pred, zero_division=0),
        "F1": f1_score(y_real, y_pred, zero_division=0),
    }
    if puntaje is not None:
        m["AUC"] = roc_auc_score(y_real, puntaje)
        m["AP"] = average_precision_score(y_real, puntaje)
    return m


def n_vectores_soporte(modelo: Pipeline) -> int:
    """Cantidad de vectores de soporte (puntos que definen el hiperplano)."""
    return int(modelo.named_steps["svm"].n_support_.sum())


def evaluar_en_prueba(modelo: Pipeline, X_te, y_te) -> dict:
    """Evalúa un SVM ya entrenado con el conjunto de prueba (datos nunca vistos)."""
    pred = modelo.predict(X_te)
    puntaje = modelo.decision_function(X_te)
    return metricas_clasificacion(y_te, pred, puntaje)


def comparar_kernels_holdout(X_tr, X_te, y_tr, y_te, class_weight="balanced",
                             parametros: dict | None = None):
    """Entrena los tres SVM con X_tr y los evalúa en X_te.

    `parametros` permite pasar hiperparámetros por kernel, p. ej.
    {"rbf": {"C": 10, "gamma": 0.1}}. Devuelve (tabla, modelos, predicciones).
    """
    parametros = parametros or {}
    filas, modelos, preds = [], {}, {}
    for nombre in KERNELS:
        modelo, seg = entrenar_svm(nombre, X_tr, y_tr, class_weight=class_weight,
                                   **parametros.get(nombre, {}))
        met = evaluar_en_prueba(modelo, X_te, y_te)
        filas.append({"Kernel": nombre, **met,
                      "Vectores_soporte": n_vectores_soporte(modelo),
                      "Seg_entrenamiento": seg})
        modelos[nombre] = modelo
        preds[nombre] = modelo.predict(X_te)
    return pd.DataFrame(filas).set_index("Kernel"), modelos, preds


def matriz_confusion(y_real, y_pred) -> pd.DataFrame:
    """Matriz de confusión con etiquetas legibles (filas = real, columnas = predicho)."""
    cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
    etiquetas = ["Normal", "Alerta"]
    return pd.DataFrame(cm, index=[f"Real: {e}" for e in etiquetas],
                        columns=[f"Pred: {e}" for e in etiquetas])
