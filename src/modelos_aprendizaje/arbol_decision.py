"""Árboles de decisión para clasificar Load_Type (Tarea 4 – DAEWOO Steel Co.).

Qué resuelve este módulo
------------------------
1. Comparar tres criterios de división: Gini, Entropía y Error de clasificación.
2. Graficar los árboles limitados a 3 y a 5 niveles de profundidad.
3. Evaluar cada árbol con el grupo de entrenamiento y el de prueba.

Nota sobre el error de clasificación
------------------------------------
scikit-learn solo ofrece "gini" y "entropy" como criterios. Para el error de
clasificación se incluye una implementación propia (`ArbolPropio`), que recibe
el mismo tipo de datos y expone `predict`, de modo que todo lo demás (métricas,
tablas, gráficos) funciona igual para los tres criterios.

Uso típico (desde el notebook)
------------------------------
    from modelos_aprendizaje import arbol_decision as ad

    tabla, modelos, preds = ad.comparar_criterios(
        X_tr, X_te, y_tr, y_te, profundidades=(3, 5),
        orden_clases=prep.ORDEN_CARGA)
    ad.graficar_arboles(modelos, list(X_tr.columns))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, recall_score)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

CRITERIOS = ("gini", "entropia", "error")
NOMBRES = {"gini": "Gini", "entropia": "Entropía", "error": "Error de clasificación"}
_CRITERIO_SKLEARN = {"gini": "gini", "entropia": "entropy"}


# ---------------------------------------------------------------------------
# 1. Medidas de impureza
# ---------------------------------------------------------------------------
def impureza(conteos, criterio: str):
    """Impureza de uno o varios nodos a partir de sus conteos por clase.

    `conteos` puede ser un vector (un nodo) o una matriz (un nodo por fila).
    Vale 0 si el nodo es puro y crece cuanto más mezcladas están las clases.
    """
    conteos = np.asarray(conteos, dtype=float)
    n = conteos.sum(axis=-1, keepdims=True)
    p = np.divide(conteos, n, out=np.zeros_like(conteos), where=n > 0)
    if criterio == "gini":
        return 1.0 - (p ** 2).sum(axis=-1)
    if criterio == "entropia":
        logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
        return -(p * logp).sum(axis=-1)
    if criterio == "error":
        return 1.0 - p.max(axis=-1)
    raise ValueError(f"Criterio desconocido: {criterio!r}. Usa {CRITERIOS}.")


# ---------------------------------------------------------------------------
# 2. Árbol propio (permite usar el error de clasificación)
# ---------------------------------------------------------------------------
class ArbolPropio:
    """Árbol de clasificación binario con umbrales sobre variables numéricas.

    En cada nodo prueba todas las preguntas "variable <= umbral" y se queda con
    la que deja la menor impureza ponderada en los dos grupos resultantes.
    Se detiene al llegar a `max_depth`, si el nodo es puro, si ninguna división
    mejora la impureza o si dejaría hojas con menos de `min_samples_leaf`.

    Con el error de clasificación es habitual que muchas divisiones no
    mejoren la impureza (la clase mayoritaria no cambia); en ese caso el árbol
    se detiene antes y su profundidad real queda por debajo de `max_depth`.
    """

    def __init__(self, criterio: str = "error", max_depth: int = 3,
                 min_samples_leaf: int = 1, min_impurity_decrease: float = 0.0):
        if criterio not in CRITERIOS:
            raise ValueError(f"Criterio desconocido: {criterio!r}.")
        self.criterio = criterio
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    # -- entrenamiento -----------------------------------------------------
    def fit(self, X, y):
        self.columnas_ = list(X.columns) if hasattr(X, "columns") else \
            [f"x{j}" for j in range(np.asarray(X).shape[1])]
        Xa = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        self._ganancias = np.zeros(Xa.shape[1])
        self.raiz_ = self._construir(Xa, yi, profundidad=0)
        total = self._ganancias.sum()
        self.feature_importances_ = (self._ganancias / total if total > 0
                                     else self._ganancias)
        return self

    def _construir(self, X, y, profundidad):
        k = len(self.classes_)
        conteos = np.bincount(y, minlength=k)
        imp = float(impureza(conteos, self.criterio))
        nodo = {"hoja": True, "n": int(len(y)), "conteos": conteos,
                "impureza": imp, "prediccion": int(conteos.argmax())}
        if profundidad >= self.max_depth or imp == 0.0:
            return nodo
        mejor = self._mejor_division(X, y, conteos, imp)
        if mejor is None:
            return nodo
        j, umbral, ganancia = mejor
        mascara = X[:, j] <= umbral
        self._ganancias[j] += ganancia * len(y)
        nodo.update(hoja=False, variable=j, umbral=float(umbral))
        nodo["izq"] = self._construir(X[mascara], y[mascara], profundidad + 1)
        nodo["der"] = self._construir(X[~mascara], y[~mascara], profundidad + 1)
        return nodo

    def _mejor_division(self, X, y, conteos_padre, imp_padre):
        n, k = X.shape[0], len(self.classes_)
        uno_hot = np.eye(k)[y]
        mejor = None  # (impureza_hijos, variable, umbral)
        for j in range(X.shape[1]):
            orden = np.argsort(X[:, j], kind="stable")
            xs = X[orden, j]
            acum = np.cumsum(uno_hot[orden], axis=0)[:-1]   # conteos a la izquierda
            n_izq = np.arange(1, n)
            validos = xs[:-1] < xs[1:]                     # solo entre valores distintos
            validos &= (n_izq >= self.min_samples_leaf) & \
                       (n - n_izq >= self.min_samples_leaf)
            if not validos.any():
                continue
            c_izq = acum[validos]
            c_der = conteos_padre - c_izq
            n_i, n_d = n_izq[validos], n - n_izq[validos]
            imp_hijos = (n_i * impureza(c_izq, self.criterio) +
                         n_d * impureza(c_der, self.criterio)) / n
            idx = int(imp_hijos.argmin())
            if mejor is None or imp_hijos[idx] < mejor[0] - 1e-12:
                pos = int(np.flatnonzero(validos)[idx])
                mejor = (float(imp_hijos[idx]), j, (xs[pos] + xs[pos + 1]) / 2)
        if mejor is None:
            return None
        ganancia = imp_padre - mejor[0]
        if ganancia <= self.min_impurity_decrease:
            return None
        return mejor[1], mejor[2], ganancia

    # -- predicción --------------------------------------------------------
    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        salida = np.empty(len(Xa), dtype=int)
        for i, fila in enumerate(Xa):
            nodo = self.raiz_
            while not nodo["hoja"]:
                nodo = nodo["izq"] if fila[nodo["variable"]] <= nodo["umbral"] \
                    else nodo["der"]
            salida[i] = nodo["prediccion"]
        return self.classes_[salida]

    # -- información del árbol --------------------------------------------
    def get_depth(self):
        def prof(nodo):
            return 0 if nodo["hoja"] else 1 + max(prof(nodo["izq"]), prof(nodo["der"]))
        return prof(self.raiz_)

    def get_n_leaves(self):
        def hojas(nodo):
            return 1 if nodo["hoja"] else hojas(nodo["izq"]) + hojas(nodo["der"])
        return hojas(self.raiz_)

    def texto(self):
        """Reglas del árbol en texto (equivalente a export_text de sklearn)."""
        lineas = []

        def recorrer(nodo, sangria):
            if nodo["hoja"]:
                lineas.append(f"{sangria}-> clase: {self.classes_[nodo['prediccion']]} "
                              f"(n={nodo['n']})")
                return
            col = self.columnas_[nodo["variable"]]
            lineas.append(f"{sangria}si {col} <= {nodo['umbral']:.3f}:")
            recorrer(nodo["izq"], sangria + "    ")
            lineas.append(f"{sangria}si {col} > {nodo['umbral']:.3f}:")
            recorrer(nodo["der"], sangria + "    ")

        recorrer(self.raiz_, "")
        return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 3. Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def entrenar_arbol(criterio: str, X_tr, y_tr, max_depth: int,
                   min_samples_leaf: int = 20, semilla: int = 42):
    """Entrena un árbol con el criterio indicado ('gini', 'entropia' o 'error').

    Gini y entropía usan scikit-learn; el error de clasificación usa `ArbolPropio`.
    Los tres comparten `max_depth` y `min_samples_leaf` para que la comparación
    sea justa (la diferencia viene solo del criterio).
    """
    if criterio in _CRITERIO_SKLEARN:
        modelo = DecisionTreeClassifier(criterion=_CRITERIO_SKLEARN[criterio],
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        random_state=semilla)
        return modelo.fit(X_tr, y_tr)
    if criterio == "error":
        return ArbolPropio("error", max_depth, min_samples_leaf).fit(X_tr, y_tr)
    raise ValueError(f"Criterio desconocido: {criterio!r}. Usa {CRITERIOS}.")


def evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te, clase_critica="Maximum_Load"):
    """Métricas de un árbol en entrenamiento y prueba. Devuelve (dict, pred_test)."""
    pred_tr, pred_te = modelo.predict(X_tr), modelo.predict(X_te)
    acc_te = accuracy_score(y_te, pred_te)
    metricas = {
        "Accuracy_train": accuracy_score(y_tr, pred_tr),
        "Accuracy_test": acc_te,
        "Error_clasif_test": 1.0 - acc_te,
        "F1_macro_test": f1_score(y_te, pred_te, average="macro"),
        f"Recall_{clase_critica}": recall_score(
            y_te, pred_te, labels=[clase_critica], average=None)[0],
        "Prof_real": modelo.get_depth(),
        "N_hojas": modelo.get_n_leaves(),
    }
    return metricas, pred_te


def comparar_criterios(X_tr, X_te, y_tr, y_te, profundidades=(3, 5),
                       criterios=CRITERIOS, min_samples_leaf=20,
                       clase_critica="Maximum_Load"):
    """Entrena un árbol por cada (criterio, profundidad) y los compara.

    Devuelve:
      tabla   -> DataFrame con las métricas de cada combinación.
      modelos -> dict {(criterio, profundidad): modelo entrenado}.
      preds   -> dict {(criterio, profundidad): predicciones en test}.
    """
    filas, modelos, preds = [], {}, {}
    for crit in criterios:
        for prof in profundidades:
            modelo = entrenar_arbol(crit, X_tr, y_tr, prof, min_samples_leaf)
            metricas, pred = evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te,
                                            clase_critica)
            filas.append({"Criterio": NOMBRES[crit], "Profundidad": prof, **metricas})
            modelos[(crit, prof)], preds[(crit, prof)] = modelo, pred
    return pd.DataFrame(filas), modelos, preds


def curva_profundidad(X_tr, X_te, y_tr, y_te, profundidades=range(1, 11),
                      criterios=CRITERIOS, min_samples_leaf=20):
    """Exactitud en train y test según la profundidad, para cada criterio.

    Sirve para ver desde qué profundidad el árbol empieza a sobreajustar
    (train sigue subiendo mientras test se estanca o baja).
    """
    filas = []
    for crit in criterios:
        for prof in profundidades:
            modelo = entrenar_arbol(crit, X_tr, y_tr, prof, min_samples_leaf)
            filas.append({"Criterio": NOMBRES[crit], "Profundidad": prof,
                          "Accuracy_train": accuracy_score(y_tr, modelo.predict(X_tr)),
                          "Accuracy_test": accuracy_score(y_te, modelo.predict(X_te))})
    return pd.DataFrame(filas)


def tabla_importancias(modelos: dict, columnas, profundidad: int):
    """Importancia de cada variable (por criterio) para una profundidad dada."""
    datos = {NOMBRES[c]: modelos[(c, p)].feature_importances_
             for (c, p) in modelos if p == profundidad}
    return pd.DataFrame(datos, index=list(columnas)).round(4)


def reglas_texto(modelo, columnas) -> str:
    """Reglas 'si ... entonces ...' del árbol, para leerlas o pegarlas en el informe."""
    if isinstance(modelo, ArbolPropio):
        return modelo.texto()
    return export_text(modelo, feature_names=list(columnas))


def verificar_arbol_propio(X_tr, X_te, y_tr, criterio="gini", max_depth=3,
                           min_samples_leaf=20):
    """Comprueba que `ArbolPropio` reproduce a scikit-learn con Gini o entropía.

    Devuelve la fracción de predicciones de test que coinciden. Debe ser
    ~1.0; pequeñas diferencias pueden venir de cómo cada implementación
    desempata divisiones equivalentes.
    """
    sk = entrenar_arbol(criterio, X_tr, y_tr, max_depth, min_samples_leaf)
    propio = ArbolPropio(criterio, max_depth, min_samples_leaf).fit(X_tr, y_tr)
    return float(np.mean(sk.predict(X_te) == propio.predict(X_te)))


# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------
def _dibujar_arbol_propio(arbol: ArbolPropio, ax):
    posiciones, contador = {}, [0]

    def ubicar(nodo, prof):
        if nodo["hoja"]:
            x = contador[0]
            contador[0] += 1
        else:
            x = (ubicar(nodo["izq"], prof + 1) + ubicar(nodo["der"], prof + 1)) / 2
        posiciones[id(nodo)] = (x, -prof)
        return x

    ubicar(arbol.raiz_, 0)
    colores = plt.cm.Pastel1(np.arange(len(arbol.classes_)))

    def dibujar(nodo):
        x, y = posiciones[id(nodo)]
        clase = arbol.classes_[nodo["prediccion"]]
        valor = [int(c) for c in nodo["conteos"]]
        base = (f"error = {nodo['impureza']:.3f}\n" if arbol.criterio == "error"
                else f"{NOMBRES[arbol.criterio]} = {nodo['impureza']:.3f}\n")
        base += f"n = {nodo['n']}\nvalor = {valor}\n{clase}"
        texto = base if nodo["hoja"] else \
            f"{arbol.columnas_[nodo['variable']]} <= {nodo['umbral']:.2f}\n{base}"
        if not nodo["hoja"]:
            for hijo in (nodo["izq"], nodo["der"]):
                xh, yh = posiciones[id(hijo)]
                ax.plot([x, xh], [y, yh], color="gray", lw=1, zorder=1)
            ax.text((x + posiciones[id(nodo["izq"])][0]) / 2, y - 0.5, "Sí",
                    fontsize=8, color="gray", ha="center")
            ax.text((x + posiciones[id(nodo["der"])][0]) / 2, y - 0.5, "No",
                    fontsize=8, color="gray", ha="center")
            dibujar(nodo["izq"])
            dibujar(nodo["der"])
        ax.text(x, y, texto, ha="center", va="center", fontsize=8, zorder=2,
                bbox=dict(boxstyle="round", fc=colores[nodo["prediccion"]], ec="gray"))

    dibujar(arbol.raiz_)
    ax.set_xlim(-0.7, max(contador[0] - 1, 0) + 0.7)
    ax.set_ylim(-arbol.get_depth() - 0.6, 0.6)
    ax.axis("off")


def graficar_arbol(modelo, columnas, titulo="", ancho_por_hoja=None, guardar=None):
    """Dibuja un árbol (de sklearn o `ArbolPropio`). Devuelve la figura."""
    hojas, prof = modelo.get_n_leaves(), modelo.get_depth()
    if ancho_por_hoja is None:  # el árbol propio lleva más texto por caja
        ancho_por_hoja = 3.0 if isinstance(modelo, ArbolPropio) else 2.4
    fig, ax = plt.subplots(figsize=(max(8, ancho_por_hoja * hojas), 2.6 * (prof + 1)))
    if isinstance(modelo, ArbolPropio):
        _dibujar_arbol_propio(modelo, ax)
    else:
        plot_tree(modelo, feature_names=list(columnas),
                  class_names=[str(c) for c in modelo.classes_], filled=True,
                  rounded=True, impurity=True, fontsize=9, ax=ax)
    ax.set_title(titulo, fontsize=14)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


def graficar_arboles(modelos: dict, columnas, carpeta_guardado=None):
    """Dibuja todos los árboles de `comparar_criterios` (criterio × profundidad)."""
    figuras = {}
    for (crit, prof), modelo in modelos.items():
        ruta = f"{carpeta_guardado}/arbol_{crit}_prof{prof}.png" if carpeta_guardado else None
        figuras[(crit, prof)] = graficar_arbol(
            modelo, columnas,
            f"Árbol de decisión – {NOMBRES[crit]} – profundidad máxima {prof} "
            f"(real: {modelo.get_depth()})", guardar=ruta)
        plt.show()
    return figuras


def graficar_curva_profundidad(tabla_curva: pd.DataFrame):
    """Una línea por criterio: train (punteada) y test (continua) vs. profundidad."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (crit, g) in enumerate(tabla_curva.groupby("Criterio", sort=False)):
        color = f"C{i}"
        ax.plot(g["Profundidad"], g["Accuracy_test"], "-o", color=color,
                label=f"{crit} – test")
        ax.plot(g["Profundidad"], g["Accuracy_train"], "--", color=color,
                alpha=0.6, label=f"{crit} – train")
    ax.set_xlabel("Profundidad máxima")
    ax.set_ylabel("Exactitud")
    ax.set_title("Exactitud según la profundidad del árbol")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def graficar_matrices_confusion(preds: dict, y_te, profundidad: int, orden_clases=None):
    """Matrices de confusión (test) de los tres criterios para una profundidad."""
    claves = [k for k in preds if k[1] == profundidad]
    fig, axes = plt.subplots(1, len(claves), figsize=(5.2 * len(claves), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (crit, prof) in zip(axes, claves):
        cm = confusion_matrix(y_te, preds[(crit, prof)], labels=orden_clases)
        ConfusionMatrixDisplay(cm, display_labels=orden_clases).plot(
            ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
        ax.set_title(f"{NOMBRES[crit]} – prof. {prof}")
    fig.tight_layout()
    return fig