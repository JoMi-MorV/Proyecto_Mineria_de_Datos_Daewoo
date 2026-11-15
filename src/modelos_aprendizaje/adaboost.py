"""AdaBoost para clasificar Load_Type (Tarea 4 – DAEWOO Steel Co.).

Idea en simple
--------------
AdaBoost entrena árboles MUY pequeños (por defecto, "tocones" de una sola pregunta)
uno tras otro. Cada árbol nuevo se concentra en los registros que los anteriores
clasificaron mal: esos registros ganan peso y los bien clasificados lo pierden.
Al final, todos los árboles votan, y el voto de cada uno vale más cuanto mejor
acertó. Es lo contrario de Random Forest: allí hay muchos árboles grandes e
independientes; aquí, muchos árboles chicos que se corrigen en cadena.

Este módulo reúne lo necesario para la Tarea 4 (parte AdaBoost): entrenar,
evaluar, elegir hiperparámetros sin mirar la prueba, ver cuántos árboles hacen
falta, medir el aporte del refuerzo y comparar todos los modelos de la tarea.

Uso típico (desde el notebook)
------------------------------
    from modelos_aprendizaje import adaboost as ab

    modelo = ab.entrenar_adaboost(X_tr, y_tr)
    metricas, pred = ab.evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

ORDEN_CARGA = ["Light_Load", "Medium_Load", "Maximum_Load"]

# Valores que se prueban al elegir hiperparámetros (se pueden cambiar al llamar).
GRILLA_POR_DEFECTO = {
    "n_estimators": [50, 200],         # cantidad de árboles pequeños
    "learning_rate": [0.1, 0.5, 1.0],  # cuánto aporta cada árbol (menor = más cauteloso)
    "max_depth": [1, 2, 3],            # profundidad de cada árbol pequeño (1 = tocón)
}


# ---------------------------------------------------------------------------
# 1. Entrenamiento
# ---------------------------------------------------------------------------
def entrenar_adaboost(X_tr, y_tr, n_estimators: int = 200, learning_rate: float = 1.0,
                      max_depth: int = 1, min_samples_leaf: int = 1,
                      semilla: int = 42):
    """Entrena un AdaBoost cuyo "árbol pequeño" tiene la profundidad indicada.

    n_estimators  -> cuántos árboles pequeños se encadenan.
    learning_rate -> peso de cada árbol en el voto final. Valores bajos exigen más
                     árboles, pero suelen sobreajustar menos.
    max_depth     -> profundidad de cada árbol pequeño (1 = una sola pregunta).
    """
    base = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  random_state=semilla)
    modelo = AdaBoostClassifier(estimator=base, n_estimators=n_estimators,
                                learning_rate=learning_rate, random_state=semilla)
    return modelo.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# 2. Evaluación
# ---------------------------------------------------------------------------
def evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te, clase_critica="Maximum_Load"):
    """Métricas en entrenamiento y prueba. Devuelve (dict, pred_test).

    Mismas métricas que en árboles y Random Forest, para poder compararlas.
    La precisión de la clase crítica indica qué fracción de lo etiquetado como
    carga máxima realmente lo es (mide las falsas alarmas).
    """
    pred_tr, pred_te = modelo.predict(X_tr), modelo.predict(X_te)
    acc_te = accuracy_score(y_te, pred_te)
    metricas = {
        "Accuracy_train": accuracy_score(y_tr, pred_tr),
        "Accuracy_test": acc_te,
        "Error_clasif_test": 1.0 - acc_te,
        "F1_macro_test": f1_score(y_te, pred_te, average="macro"),
        f"Recall_{clase_critica}": float(recall_score(
            y_te, pred_te, labels=[clase_critica], average=None)[0]),
        f"Precision_{clase_critica}": float(precision_score(
            y_te, pred_te, labels=[clase_critica], average=None, zero_division=0)[0]),
    }
    return metricas, pred_te


# ---------------------------------------------------------------------------
# 3. Elección de hiperparámetros (sin mirar el grupo de prueba)
# ---------------------------------------------------------------------------
def buscar_hiperparametros(X_tr, y_tr, grilla: dict | None = None, n_splits: int = 5,
                           semilla: int = 42, n_jobs: int = -1):
    """Prueba combinaciones con validación cruzada temporal dentro del entrenamiento.

    Se entrena con los primeros meses y se valida con los siguientes
    (TimeSeriesSplit), así el grupo de prueba queda intacto. El criterio es el
    F1 macro. Devuelve (mejores_parametros, tabla ordenada de mejor a peor).
    """
    grilla = grilla or GRILLA_POR_DEFECTO
    grilla_sk = {("estimator__max_depth" if k == "max_depth" else k): v
                 for k, v in grilla.items()}
    busqueda = GridSearchCV(
        AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=semilla),
                           random_state=semilla),
        param_grid=grilla_sk, scoring="f1_macro", cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=n_jobs, refit=False)
    # Se pasan arreglos (no DataFrame) para evitar advertencias de nombres de columnas
    # que scikit-learn emite al paralelizar el refuerzo.
    busqueda.fit(np.asarray(X_tr), np.asarray(y_tr))
    r = pd.DataFrame(busqueda.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")]
    tabla = r[cols + ["mean_test_score", "std_test_score", "rank_test_score"]].copy()
    tabla.columns = [c.replace("param_", "").replace("estimator__", "") for c in cols] + \
                    ["F1_macro_CV", "Desv_CV", "Ranking"]
    tabla = tabla.sort_values("Ranking").reset_index(drop=True)
    mejores = {k.replace("estimator__", ""): v for k, v in busqueda.best_params_.items()}
    return mejores, tabla


# ---------------------------------------------------------------------------
# 4. Cuántos árboles hacen falta
# ---------------------------------------------------------------------------
def curva_n_estimadores(X_tr, X_te, y_tr, y_te, n_estimators: int = 300,
                        learning_rate: float = 1.0, max_depth: int = 1,
                        paso: int = 10, semilla: int = 42):
    """Exactitud en train y test según el número de árboles encadenados.

    Se entrena una sola vez y se evalúa el modelo "hasta la ronda k" (staged_predict),
    por lo que es rápido. Sirve para ver cuándo seguir agregando árboles deja de ayudar.
    """
    modelo = entrenar_adaboost(X_tr, y_tr, n_estimators, learning_rate, max_depth,
                               semilla=semilla)
    total = len(modelo.estimators_)
    filas = []
    for k, (p_tr, p_te) in enumerate(zip(modelo.staged_predict(X_tr),
                                         modelo.staged_predict(X_te)), start=1):
        if k == 1 or k % paso == 0 or k == total:
            filas.append({"N_estimadores": k,
                          "Accuracy_train": accuracy_score(y_tr, p_tr),
                          "Accuracy_test": accuracy_score(y_te, p_te),
                          "F1_macro_test": f1_score(y_te, p_te, average="macro")})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 5. Comparaciones
# ---------------------------------------------------------------------------
def comparar_base_vs_refuerzo(X_tr, X_te, y_tr, y_te, params_ab: dict | None = None,
                              clase_critica="Maximum_Load", semilla: int = 42):
    """Compara el árbol pequeño solo con el AdaBoost que lo usa de base.

    Responde: ¿encadenar muchos árboles pequeños mejora al árbol pequeño aislado?
    Devuelve (tabla, modelos, preds).
    """
    params_ab = dict(params_ab or {})
    prof = params_ab.get("max_depth", 1)
    base = DecisionTreeClassifier(max_depth=prof, random_state=semilla).fit(X_tr, y_tr)
    refuerzo = entrenar_adaboost(X_tr, y_tr, semilla=semilla, **params_ab)
    modelos = {f"Árbol pequeño solo (prof. {prof})": base, "AdaBoost": refuerzo}
    filas, preds = [], {}
    for nombre, m in modelos.items():
        metricas, pred = evaluar_modelo(m, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Modelo": nombre, **metricas})
        preds[nombre] = pred
    return pd.DataFrame(filas), modelos, preds


def comparar_aprendices_debiles(X_tr, X_te, y_tr, y_te, profundidades=(1, 2, 3, 5),
                                n_estimators: int = 200, learning_rate: float = 0.5,
                                clase_critica="Maximum_Load"):
    """AdaBoost con árboles pequeños de distinta profundidad (1 = tocón).

    Muestra si conviene que cada árbol del encadenamiento sea más o menos complejo.
    """
    filas = []
    for prof in profundidades:
        m = entrenar_adaboost(X_tr, y_tr, n_estimators, learning_rate, prof)
        metricas, _ = evaluar_modelo(m, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Profundidad_base": prof, **metricas})
    return pd.DataFrame(filas)


def comparar_modelos(modelos: dict, X_tr, X_te, y_tr, y_te, clase_critica="Maximum_Load"):
    """Tabla final para comparar CUALQUIER modelo con método `predict`.

    `modelos` es un dict {nombre: modelo ya entrenado}. Sirve para juntar el árbol,
    el Random Forest, AdaBoost y la red bayesiana en una sola tabla, con las mismas
    métricas. Devuelve (tabla, preds).
    """
    filas, preds = [], {}
    for nombre, m in modelos.items():
        metricas, pred = evaluar_modelo(m, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Modelo": nombre, **metricas})
        preds[nombre] = pred
    return pd.DataFrame(filas), preds


# ---------------------------------------------------------------------------
# 6. Importancia de variables
# ---------------------------------------------------------------------------
def tabla_importancias(modelo, X_te, y_te, n_repeticiones: int = 5, semilla: int = 42,
                       n_jobs: int = -1):
    """Qué variables pesan más (por impureza y por permutación en prueba).

    La de permutación mide cuánto baja el F1 macro en PRUEBA al desordenar una
    variable; es más confiable porque se mide con datos no vistos.
    """
    perm = permutation_importance(modelo, X_te, y_te, scoring="f1_macro",
                                  n_repeats=n_repeticiones, random_state=semilla,
                                  n_jobs=n_jobs)
    tabla = pd.DataFrame({"Imp_impureza": modelo.feature_importances_,
                          "Imp_permutacion": perm.importances_mean,
                          "Desv_permutacion": perm.importances_std},
                         index=list(X_te.columns))
    return tabla.sort_values("Imp_permutacion", ascending=False).round(4)


# ---------------------------------------------------------------------------
# 7. Gráficos
# ---------------------------------------------------------------------------
def graficar_curva_n_estimadores(tabla_curva: pd.DataFrame):
    """Exactitud en train (punteada) y test (continua) según las rondas de refuerzo."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(tabla_curva["N_estimadores"], tabla_curva["Accuracy_test"], "-o", ms=3,
            label="Test")
    ax.plot(tabla_curva["N_estimadores"], tabla_curva["Accuracy_train"], "--", alpha=0.7,
            label="Train")
    ax.set_xlabel("Número de árboles encadenados")
    ax.set_ylabel("Exactitud")
    ax.set_title("AdaBoost: exactitud según el número de árboles")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def graficar_rondas(modelo):
    """Error ponderado y peso de voto de cada árbol, ronda por ronda.

    Muestra cómo funciona el refuerzo: los árboles posteriores enfrentan los casos
    más difíciles, por eso su error sube y su peso en el voto final baja.
    """
    errores = modelo.estimator_errors_[:len(modelo.estimators_)]
    pesos = modelo.estimator_weights_[:len(modelo.estimators_)]
    rondas = np.arange(1, len(errores) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(rondas, errores, color="C3")
    axes[0].set_title("Error de cada árbol (ponderado)")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("Error")
    axes[1].plot(rondas, pesos, color="C2")
    axes[1].set_title("Peso de cada árbol en el voto final")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("Peso")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graficar_importancias(tabla: pd.DataFrame):
    """Barras horizontales: importancia por permutación y por impureza."""
    t = tabla.sort_values("Imp_permutacion")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].barh(t.index, t["Imp_permutacion"], xerr=t["Desv_permutacion"], color="C0")
    axes[0].set_title("Por permutación (en prueba)")
    axes[0].set_xlabel("Caída del F1 macro")
    axes[1].barh(t.index, t["Imp_impureza"], color="C1")
    axes[1].set_title("Por impureza (en entrenamiento)")
    axes[1].set_xlabel("Importancia relativa")
    for ax in axes:
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Importancia de las variables en AdaBoost")
    fig.tight_layout()
    return fig


def graficar_matrices_confusion(preds: dict, y_te, orden_clases=ORDEN_CARGA):
    """Matrices de confusión (test) de uno o varios modelos, lado a lado."""
    fig, axes = plt.subplots(1, len(preds), figsize=(5.4 * len(preds), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (nombre, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_te, pred, labels=orden_clases)
        ConfusionMatrixDisplay(cm, display_labels=orden_clases).plot(
            ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
        ax.set_title(nombre)
    fig.tight_layout()
    return fig