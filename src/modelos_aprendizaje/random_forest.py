"""Random Forest para clasificar Load_Type (Tarea 4 – DAEWOO Steel Co.).

Idea en simple
--------------
Un Random Forest (bosque aleatorio) entrena MUCHOS árboles de decisión, cada uno
con una versión ligeramente distinta de los datos (una muestra al azar de los
registros y, en cada pregunta, solo un subconjunto al azar de las variables).
Para predecir, cada árbol "vota" por un nivel de carga y gana el más votado.
Como los árboles se equivocan en cosas distintas, al combinarlos los errores
tienden a compensarse y el resultado suele ser más estable que el de un solo árbol.

Este módulo reúne todo lo necesario para la Tarea 4 (parte Random Forest):
entrenar, evaluar, elegir hiperparámetros sin mirar la prueba, ver cuántos
árboles hacen falta, medir qué variables importan y comparar con un árbol único.

Uso típico (desde el notebook)
------------------------------
    from modelos_aprendizaje import random_forest as rf

    modelo = rf.entrenar_random_forest(X_tr, y_tr)
    metricas, pred = rf.evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree

ORDEN_CARGA = ["Light_Load", "Medium_Load", "Maximum_Load"]

# Valores que se prueban al elegir hiperparámetros (se pueden cambiar al llamar).
GRILLA_POR_DEFECTO = {
    "n_estimators": [200],
    "max_depth": [5, 10, None],        # None = sin límite de profundidad
    "min_samples_leaf": [5, 20, 50],
    "max_features": ["sqrt", 0.5],     # variables que mira cada árbol en cada pregunta
}


# ---------------------------------------------------------------------------
# 1. Entrenamiento
# ---------------------------------------------------------------------------
def entrenar_random_forest(X_tr, y_tr, n_estimators: int = 100, max_depth=None,
                           max_features="sqrt", min_samples_leaf: int = 20,
                           class_weight=None, semilla: int = 42, n_jobs: int = -1,
                           **extra):
    """Entrena un Random Forest con los datos de entrenamiento.

    n_estimators     -> cantidad de árboles del bosque.
    max_depth        -> profundidad máxima de cada árbol (None = sin límite).
    max_features     -> variables que considera cada árbol en cada pregunta
                        ("sqrt" = raíz cuadrada del total, 0.5 = la mitad, None = todas).
    min_samples_leaf -> mínimo de registros por hoja (evita memorizar lecturas casi idénticas).
    class_weight     -> None, o "balanced" para dar más peso a las clases menos frecuentes.
    """
    modelo = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        min_samples_leaf=min_samples_leaf, class_weight=class_weight,
        random_state=semilla, n_jobs=n_jobs, **extra)
    return modelo.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# 2. Evaluación
# ---------------------------------------------------------------------------
def evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te, clase_critica="Maximum_Load"):
    """Métricas del modelo en entrenamiento y prueba. Devuelve (dict, pred_test).

    Incluye las mismas métricas que en los árboles (para poder compararlas) más la
    precisión de la clase crítica: de todo lo que el modelo etiqueta como carga
    máxima, qué fracción realmente lo es (mide las "falsas alarmas").
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
    """Prueba combinaciones de hiperparámetros con validación cruzada temporal.

    Se usa solo el grupo de entrenamiento: se entrena con los primeros meses y se
    valida con los siguientes (TimeSeriesSplit), y así el grupo de prueba queda
    intacto para la evaluación final. El criterio es el F1 macro.

    Devuelve (mejores_parametros, tabla), con la tabla ordenada de mejor a peor.
    Atención: puede tardar unos minutos con la grilla por defecto.
    """
    grilla = grilla or GRILLA_POR_DEFECTO
    busqueda = GridSearchCV(
        RandomForestClassifier(random_state=semilla, n_jobs=1),
        param_grid=grilla, scoring="f1_macro", cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=n_jobs, refit=False)
    busqueda.fit(X_tr, y_tr)
    r = pd.DataFrame(busqueda.cv_results_)
    tabla = r[[c for c in r.columns if c.startswith("param_")] +
              ["mean_test_score", "std_test_score", "rank_test_score"]].copy()
    tabla.columns = [c.replace("param_", "") for c in tabla.columns[:-3]] + \
                    ["F1_macro_CV", "Desv_CV", "Ranking"]
    tabla = tabla.sort_values("Ranking").reset_index(drop=True)
    return busqueda.best_params_, tabla


# ---------------------------------------------------------------------------
# 4. Cuántos árboles hacen falta
# ---------------------------------------------------------------------------
def curva_n_arboles(X_tr, X_te, y_tr, y_te, valores=(1, 5, 10, 25, 50, 100, 200, 300),
                    max_depth=None, max_features="sqrt", min_samples_leaf=20,
                    semilla: int = 42, n_jobs: int = -1):
    """Exactitud en train y test según el número de árboles del bosque.

    Reutiliza los árboles ya entrenados al ir aumentando la cantidad (warm_start),
    por lo que es rápido. Sirve para ver desde qué cantidad agregar árboles ya no mejora.
    """
    modelo = RandomForestClassifier(
        n_estimators=valores[0], max_depth=max_depth, max_features=max_features,
        min_samples_leaf=min_samples_leaf, warm_start=True, random_state=semilla,
        n_jobs=n_jobs)
    filas = []
    for n in valores:
        modelo.set_params(n_estimators=n)
        modelo.fit(X_tr, y_tr)
        pred_te = modelo.predict(X_te)
        filas.append({"N_arboles": n,
                      "Accuracy_train": accuracy_score(y_tr, modelo.predict(X_tr)),
                      "Accuracy_test": accuracy_score(y_te, pred_te),
                      "F1_macro_test": f1_score(y_te, pred_te, average="macro")})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 5. Comparaciones
# ---------------------------------------------------------------------------
def comparar_bosque_vs_arbol(X_tr, X_te, y_tr, y_te, params_rf: dict | None = None,
                             profundidad_arbol: int = 5, min_samples_leaf: int = 20,
                             clase_critica="Maximum_Load", semilla: int = 42):
    """Compara un árbol único (Gini) con el Random Forest, en las mismas condiciones.

    Responde: ¿combinar muchos árboles mejora a uno solo? Devuelve (tabla, modelos, preds).
    """
    params_rf = dict(params_rf or {})
    arbol = DecisionTreeClassifier(criterion="gini", max_depth=profundidad_arbol,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=semilla).fit(X_tr, y_tr)
    bosque = entrenar_random_forest(X_tr, y_tr, semilla=semilla, **params_rf)
    modelos = {f"Árbol único (Gini, prof. {profundidad_arbol})": arbol,
               "Random Forest": bosque}
    filas, preds = [], {}
    for nombre, m in modelos.items():
        metricas, pred = evaluar_modelo(m, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Modelo": nombre, **metricas})
        preds[nombre] = pred
    return pd.DataFrame(filas), modelos, preds


def comparar_class_weight(X_tr, X_te, y_tr, y_te, params_rf: dict | None = None,
                          clase_critica="Maximum_Load"):
    """Compara el bosque normal con uno que da más peso a las clases menos frecuentes.

    Pregunta: ¿se puede detectar más carga máxima? Normalmente sube la sensibilidad
    de la clase crítica, pero baja su precisión (más falsas alarmas).
    """
    params_rf = {k: v for k, v in (params_rf or {}).items() if k != "class_weight"}
    filas = []
    for etiqueta, cw in (("Sin ponderar", None), ("Ponderado (balanced)", "balanced")):
        m = entrenar_random_forest(X_tr, y_tr, class_weight=cw, **params_rf)
        metricas, _ = evaluar_modelo(m, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Versión": etiqueta, **metricas})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 6. Importancia de variables
# ---------------------------------------------------------------------------
def tabla_importancias(modelo, X_te, y_te, n_repeticiones: int = 5, semilla: int = 42,
                       n_jobs: int = -1):
    """Qué variables pesan más en las decisiones del bosque (de dos formas).

    - Imp_impureza: cuánto reduce cada variable la impureza al dividir (se calcula
      al entrenar; puede favorecer variables continuas).
    - Imp_permutacion: cuánto baja el F1 macro en PRUEBA al desordenar al azar esa
      variable. Es más confiable porque mide el efecto sobre datos no vistos.
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
def graficar_curva_n_arboles(tabla_curva: pd.DataFrame):
    """Exactitud en train (punteada) y test (continua) según el número de árboles."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(tabla_curva["N_arboles"], tabla_curva["Accuracy_test"], "-o", label="Test")
    ax.plot(tabla_curva["N_arboles"], tabla_curva["Accuracy_train"], "--", alpha=0.7,
            label="Train")
    ax.set_xscale("log")
    ax.set_xlabel("Número de árboles (escala logarítmica)")
    ax.set_ylabel("Exactitud")
    ax.set_title("Random Forest: exactitud según el número de árboles")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def graficar_importancias(tabla: pd.DataFrame):
    """Barras horizontales con la importancia por permutación y por impureza."""
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
    fig.suptitle("Importancia de las variables en el Random Forest")
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


def graficar_arbol_del_bosque(modelo, columnas, indice: int = 0, max_depth: int = 3):
    """Dibuja UNO de los árboles del bosque (los primeros `max_depth` niveles).

    Sirve para mostrar que el bosque está hecho de árboles como los de la sección
    anterior, pero cada uno entrenado con una muestra distinta de datos y variables.
    """
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(modelo.estimators_[indice], feature_names=list(columnas),
              class_names=[str(c) for c in modelo.classes_], filled=True, rounded=True,
              max_depth=max_depth, fontsize=9, ax=ax)
    ax.set_title(f"Árbol n.º {indice} del Random Forest (primeros {max_depth} niveles)")
    fig.tight_layout()
    return fig