"""Redes Bayesianas para el nivel de carga de la planta (Tarea 4 – DAEWOO Steel Co.).

Idea en simple
--------------
Una red bayesiana es un diagrama de flechas entre variables. Cada flecha dice
"esta variable influye directamente en esa otra", y cada variable tiene una tabla
de probabilidades que depende de las variables que apuntan hacia ella (sus
"padres"). Con esa red se puede responder preguntas de probabilidad, por ejemplo:
"si es de tarde, día hábil y el factor de potencia es bajo, ¿qué probabilidad hay
de carga máxima?". Es el único modelo de la tarea que entrega probabilidades
explícitas y que funciona aunque solo se conozca parte de la información.

Pasos de este módulo
--------------------
1. Discretizar: las redes de este tipo trabajan con categorías, no con números
   continuos. Las variables se agrupan en pocos niveles (Bajo/Medio/Alto, franjas
   horarias, etc.) usando solo los datos de entrenamiento.
2. Definir la estructura (las flechas): Naive Bayes, una estructura experta
   definida con criterio del problema y una aprendida de los datos (Hill Climbing
   con puntaje BIC).
3. Estimar las tablas de probabilidad con los datos de entrenamiento.
4. Predecir `Load_Type` y evaluar con entrenamiento y prueba.
5. Hacer consultas de probabilidad con evidencia parcial.

Nota técnica: está implementada solo con numpy y pandas (sin pgmpy), con
inferencia exacta por enumeración, que es suficiente con 7 variables.

Uso típico (desde el notebook)
------------------------------
    from modelos_aprendizaje import redes_bayesianas as rb

    tabla, redes, preds = rb.comparar_estructuras(X_tr, X_te, y_tr, y_te)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

ORDEN_CARGA = ["Light_Load", "Medium_Load", "Maximum_Load"]

# Nodos de la red (el primero es la variable a predecir)
NODO_CLASE = "Load_Type"
N_HORA = "Franja_horaria"
N_DIA = "Tipo_dia"
N_LAGK = "Reactiva_inductiva"
N_LEADK = "Reactiva_capacitiva"
N_LAGPF = "FP_inductivo"
N_LEADPF = "FP_capacitivo"
NODOS = [NODO_CLASE, N_HORA, N_DIA, N_LAGK, N_LEADK, N_LAGPF, N_LEADPF]

# Qué columna numérica origina cada nodo continuo
COLUMNAS_CONTINUAS = {
    N_LAGK: "Lagging_Current_Reactive.Power_kVarh",
    N_LEADK: "Leading_Current_Reactive_Power_kVarh",
    N_LAGPF: "Lagging_Current_Power_Factor",
    N_LEADPF: "Leading_Current_Power_Factor",
}
FRANJAS = ["Madrugada", "Mañana", "Tarde", "Noche"]   # 0-6, 6-12, 12-18, 18-24 h
DIAS = ["Hábil", "Fin de semana"]

# Estructuras con flechas definidas a mano (padre, hijo)
ESTRUCTURAS_FIJAS = {
    # Naive Bayes: el nivel de carga "produce" todas las demás variables
    "naive_bayes": [(NODO_CLASE, n) for n in NODOS[1:]],
    # Experta: el horario y el tipo de día determinan la carga; la carga determina las
    # mediciones eléctricas; y cada factor de potencia depende de su potencia reactiva.
    "experta": [(N_HORA, NODO_CLASE), (N_DIA, NODO_CLASE),
                (NODO_CLASE, N_LAGK), (NODO_CLASE, N_LEADK),
                (NODO_CLASE, N_LAGPF), (NODO_CLASE, N_LEADPF),
                (N_LAGK, N_LAGPF), (N_LEADK, N_LEADPF)],
}

_NOMBRES_NIVELES = {
    1: ["Único"], 2: ["Bajo", "Alto"], 3: ["Bajo", "Medio", "Alto"],
    4: ["Muy bajo", "Bajo", "Alto", "Muy alto"],
    5: ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"],
}
_NOMBRES_VISIBLES = {
    NODO_CLASE: "Load_Type", N_HORA: "Franja\nhoraria", N_DIA: "Tipo\nde día",
    N_LAGK: "Reactiva\ninductiva", N_LEADK: "Reactiva\ncapacitiva",
    N_LAGPF: "Factor pot.\ninductivo", N_LEADPF: "Factor pot.\ncapacitivo",
}


# ---------------------------------------------------------------------------
# 1. Utilidades de datos
# ---------------------------------------------------------------------------
def hora_del_dia(X) -> pd.Series:
    """Hora del día (0 a 24) a partir de NSM_sin y NSM_cos (o de NSM si existe).

    Conviene verificarla contra los datos originales: debe coincidir con
    `df.loc[X.index, "NSM"] / 3600`.
    """
    if "NSM" in X.columns:
        return X["NSM"] / 3600.0
    ang = np.arctan2(X["NSM_sin"].to_numpy(), X["NSM_cos"].to_numpy()) % (2 * np.pi)
    return pd.Series(ang / (2 * np.pi) * 24.0, index=X.index, name="hora")


def _nombres_niveles(k: int):
    return _NOMBRES_NIVELES.get(k, [f"Nivel {i + 1}" for i in range(k)])


# ---------------------------------------------------------------------------
# 2. Puntaje BIC y búsqueda de estructura (Hill Climbing)
# ---------------------------------------------------------------------------
def _bic_nodo(D, card, j, padres):
    """Puntaje BIC de un nodo dado sus padres (mayor = mejor).

    BIC = verosimilitud de los datos - penalización por complejidad (número de
    parámetros). Premia las flechas que realmente explican los datos y castiga las
    que solo agregan parámetros.
    """
    n, r = len(D), card[j]
    if padres:
        idx = np.ravel_multi_index([D[:, p] for p in padres], [card[p] for p in padres])
    else:
        idx = np.zeros(n, dtype=int)
    q = int(np.prod([card[p] for p in padres])) if padres else 1
    cuentas = np.bincount(idx * r + D[:, j], minlength=q * r).reshape(q, r).astype(float)
    tot = cuentas.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(cuentas > 0, cuentas * np.log(cuentas / tot), 0.0).sum()
    return ll - 0.5 * np.log(n) * q * (r - 1)


def _hay_camino(padres, origen, destino):
    """¿Existe un camino dirigido origen -> ... -> destino?"""
    hijos = {i: [h for h in padres if i in padres[h]] for i in padres}
    pila, visto = [origen], set()
    while pila:
        x = pila.pop()
        if x == destino:
            return True
        if x in visto:
            continue
        visto.add(x)
        pila.extend(hijos[x])
    return False


def _hill_climbing(D, card, max_indegree, prohibidos_destino):
    """Parte sin flechas y, en cada paso, aplica la mejor jugada (agregar, quitar o
    invertir una flecha) mientras el puntaje BIC siga mejorando. Evita ciclos."""
    n_nodos = len(card)
    padres = {i: set() for i in range(n_nodos)}
    cache = {}

    def pts(j, ps):
        clave = (j, tuple(sorted(ps)))
        if clave not in cache:
            cache[clave] = _bic_nodo(D, card, j, sorted(ps))
        return cache[clave]

    while True:
        mejor_delta, mejor_op = 1e-9, None
        for a in range(n_nodos):
            for b in range(n_nodos):
                if a == b:
                    continue
                actual_b = pts(b, padres[b])
                if a not in padres[b]:
                    # Agregar a -> b
                    if (b in prohibidos_destino or len(padres[b]) >= max_indegree
                            or _hay_camino(padres, b, a)):
                        continue
                    delta = pts(b, padres[b] | {a}) - actual_b
                    if delta > mejor_delta:
                        mejor_delta, mejor_op = delta, ("agregar", a, b)
                else:
                    # Quitar a -> b
                    delta = pts(b, padres[b] - {a}) - actual_b
                    if delta > mejor_delta:
                        mejor_delta, mejor_op = delta, ("quitar", a, b)
                    # Invertir a -> b (queda b -> a)
                    if a in prohibidos_destino or len(padres[a]) >= max_indegree:
                        continue
                    sin = {k: (v - {a} if k == b else set(v)) for k, v in padres.items()}
                    if _hay_camino(sin, a, b):
                        continue
                    delta = (pts(b, padres[b] - {a}) - actual_b
                             + pts(a, padres[a] | {b}) - pts(a, padres[a]))
                    if delta > mejor_delta:
                        mejor_delta, mejor_op = delta, ("invertir", a, b)
        if mejor_op is None:
            break
        op, a, b = mejor_op
        if op == "agregar":
            padres[b].add(a)
        elif op == "quitar":
            padres[b].discard(a)
        else:
            padres[b].discard(a)
            padres[a].add(b)
    return {j: sorted(p) for j, p in padres.items()}


# ---------------------------------------------------------------------------
# 3. La red bayesiana
# ---------------------------------------------------------------------------
class RedBayesiana:
    """Red bayesiana discreta para predecir `Load_Type`.

    Recibe las mismas variables numéricas que los demás modelos (con NSM_sin y
    NSM_cos) y se encarga de discretizarlas internamente con cortes calculados
    solo con los datos de entrenamiento. Tiene `fit`, `predict` y `predict_proba`
    como los modelos de scikit-learn.

    estructura: "naive_bayes", "experta", "aprendida" o una lista de pares
                (padre, hijo) con los nombres de nodo de `NODOS`.
    n_bins: niveles en que se divide cada variable continua (por cuantiles).
    max_indegree: máximo de padres por nodo en la estructura aprendida.
    prohibidos_destino: nodos que no pueden recibir flechas al aprender la
                estructura (el horario y el tipo de día no son "causados" por la carga).
    alpha: suavizado de las probabilidades (evita probabilidades exactamente 0).
    valor_finde: valor de WeekStatus_bin que significa fin de semana. "auto" toma
                el menos frecuente (los fines de semana son ~2/7 de los días).
    """

    def __init__(self, estructura="aprendida", n_bins: int = 3, max_indegree: int = 3,
                 prohibidos_destino=(N_HORA, N_DIA), alpha: float = 1.0,
                 valor_finde="auto"):
        self.estructura = estructura
        self.n_bins = n_bins
        self.max_indegree = max_indegree
        self.prohibidos_destino = tuple(prohibidos_destino)
        self.alpha = alpha
        self.valor_finde = valor_finde

    # -- discretización ----------------------------------------------------
    def _ajustar_discretizacion(self, X):
        self.cortes_ = {}
        for nodo, col in COLUMNAS_CONTINUAS.items():
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.cortes_[nodo] = np.unique(np.quantile(X[col].to_numpy(float), qs))
        if self.valor_finde == "auto":
            vc = X["WeekStatus_bin"].value_counts()
            self.valor_finde_ = vc.idxmin() if len(vc) > 1 else 1
        else:
            self.valor_finde_ = self.valor_finde

    def _discretizar(self, X) -> pd.DataFrame:
        d = pd.DataFrame(index=X.index)
        hora = hora_del_dia(X).to_numpy()
        d[N_HORA] = np.digitize(hora % 24, [6, 12, 18])
        d[N_DIA] = (X["WeekStatus_bin"].to_numpy() == self.valor_finde_).astype(int)
        for nodo, col in COLUMNAS_CONTINUAS.items():
            d[nodo] = np.searchsorted(self.cortes_[nodo], X[col].to_numpy(float), side="left")
        return d[NODOS[1:]]

    # -- entrenamiento -----------------------------------------------------
    def fit(self, X, y):
        y = pd.Series(np.asarray(y))
        self._ajustar_discretizacion(X)
        Xd = self._discretizar(X)
        presentes = set(y.unique())
        self.clases_ = [c for c in ORDEN_CARGA if c in presentes] + \
                       sorted(presentes - set(ORDEN_CARGA))
        self.classes_ = np.array(self.clases_)
        mapa = {c: i for i, c in enumerate(self.clases_)}
        D = np.column_stack([y.map(mapa).to_numpy()] +
                            [Xd[n].to_numpy() for n in NODOS[1:]]).astype(int)
        self.estados_ = {NODO_CLASE: list(self.clases_), N_HORA: list(FRANJAS),
                         N_DIA: list(DIAS)}
        for nodo in COLUMNAS_CONTINUAS:
            self.estados_[nodo] = _nombres_niveles(len(self.cortes_[nodo]) + 1)
        self.card_ = [len(self.estados_[n]) for n in NODOS]

        # Estructura
        if isinstance(self.estructura, str) and self.estructura == "aprendida":
            prohibidos = {NODOS.index(n) for n in self.prohibidos_destino}
            self.padres_ = _hill_climbing(D, self.card_, self.max_indegree, prohibidos)
        else:
            arcos = ESTRUCTURAS_FIJAS[self.estructura] if isinstance(self.estructura, str) \
                else self.estructura
            padres = {j: [] for j in range(len(NODOS))}
            for p, h in arcos:
                padres[NODOS.index(h)].append(NODOS.index(p))
            self.padres_ = {j: sorted(ps) for j, ps in padres.items()}

        # Tablas de probabilidad condicional (con suavizado de Laplace)
        self.cpt_ = {}
        for j in range(len(NODOS)):
            r = self.card_[j]
            ps = self.padres_[j]
            q = int(np.prod([self.card_[p] for p in ps])) if ps else 1
            idx = self._indice_padres(D, j)
            cuentas = np.bincount(idx * r + D[:, j], minlength=q * r).reshape(q, r)
            cuentas = cuentas.astype(float) + self.alpha
            self.cpt_[j] = cuentas / cuentas.sum(axis=1, keepdims=True)

        self.bic_ = float(sum(_bic_nodo(D, self.card_, j, self.padres_[j])
                              for j in range(len(NODOS))))
        self.n_parametros_ = int(sum(
            (int(np.prod([self.card_[p] for p in self.padres_[j]])) if self.padres_[j] else 1)
            * (self.card_[j] - 1) for j in range(len(NODOS))))
        return self

    # -- inferencia --------------------------------------------------------
    def _indice_padres(self, D, j):
        ps = self.padres_[j]
        if not ps:
            return np.zeros(len(D), dtype=int)
        return np.ravel_multi_index([D[:, p] for p in ps], [self.card_[p] for p in ps])

    def _log_conjunta(self, D):
        """log P(todas las variables) = suma de log P(nodo | padres) de cada nodo."""
        lp = np.zeros(len(D))
        for j in range(len(NODOS)):
            lp += np.log(self.cpt_[j][self._indice_padres(D, j), D[:, j]])
        return lp

    def predict_proba(self, X) -> pd.DataFrame:
        """Probabilidad de cada nivel de carga para cada registro."""
        Xd = self._discretizar(X)
        D = np.column_stack([np.zeros(len(Xd), dtype=int)] +
                            [Xd[n].to_numpy() for n in NODOS[1:]]).astype(int)
        logp = np.empty((len(D), len(self.clases_)))
        for c in range(len(self.clases_)):
            D[:, 0] = c
            logp[:, c] = self._log_conjunta(D)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        return pd.DataFrame(p, index=X.index, columns=self.clases_)

    def predict(self, X):
        """Nivel de carga más probable para cada registro."""
        return self.classes_[self.predict_proba(X).to_numpy().argmax(axis=1)]

    def consulta(self, evidencia: dict, objetivo: str = NODO_CLASE) -> pd.Series:
        """P(objetivo | evidencia): inferencia exacta por enumeración.

        `evidencia` es un dict {nodo: estado}, por ejemplo
        {"Franja_horaria": "Tarde", "Tipo_dia": "Hábil"}. Las variables que no se
        mencionan se promedian (se suman sobre todos sus valores posibles).
        """
        if objetivo not in NODOS:
            raise ValueError(f"Nodo desconocido: {objetivo!r}. Nodos: {NODOS}")
        if objetivo in evidencia:
            raise ValueError("El objetivo no puede estar también en la evidencia.")
        fijos = {}
        for nodo, estado in evidencia.items():
            if nodo not in NODOS:
                raise ValueError(f"Nodo desconocido: {nodo!r}. Nodos: {NODOS}")
            if estado not in self.estados_[nodo]:
                raise ValueError(f"Estado {estado!r} inválido para {nodo!r}. "
                                 f"Estados: {self.estados_[nodo]}")
            fijos[NODOS.index(nodo)] = self.estados_[nodo].index(estado)
        libres = [j for j in range(len(NODOS)) if j not in fijos]
        rejilla = np.indices([self.card_[j] for j in libres]).reshape(len(libres), -1).T
        A = np.zeros((len(rejilla), len(NODOS)), dtype=int)
        for j, v in fijos.items():
            A[:, j] = v
        for k, j in enumerate(libres):
            A[:, j] = rejilla[:, k]
        lp = self._log_conjunta(A)
        p = np.exp(lp - lp.max())
        j_obj = NODOS.index(objetivo)
        res = np.bincount(A[:, j_obj], weights=p, minlength=self.card_[j_obj])
        return pd.Series(res / res.sum(), index=self.estados_[objetivo],
                         name=f"P({objetivo} | evidencia)")

    # -- información de la red --------------------------------------------
    @property
    def arcos_(self):
        """Lista de flechas (padre, hijo) de la red."""
        return [(NODOS[p], NODOS[h]) for h, ps in self.padres_.items() for p in ps]


# ---------------------------------------------------------------------------
# 4. Tablas para el informe
# ---------------------------------------------------------------------------
def tabla_cortes(red: RedBayesiana) -> pd.DataFrame:
    """Cómo se discretizó cada variable: niveles y valores de corte."""
    filas = [{"Variable": N_HORA, "Origen": "NSM_sin / NSM_cos",
              "Niveles": "Madrugada 0-6 h | Mañana 6-12 h | Tarde 12-18 h | Noche 18-24 h"},
             {"Variable": N_DIA, "Origen": "WeekStatus_bin",
              "Niveles": f"Hábil | Fin de semana (valor {red.valor_finde_})"}]
    for nodo, col in COLUMNAS_CONTINUAS.items():
        cortes, nombres = red.cortes_[nodo], red.estados_[nodo]
        partes = []
        for i, nom in enumerate(nombres):
            if len(cortes) == 0:
                partes.append(nom)
            elif i == 0:
                partes.append(f"{nom}: ≤ {cortes[0]:.2f}")
            elif i == len(cortes):
                partes.append(f"{nom}: > {cortes[-1]:.2f}")
            else:
                partes.append(f"{nom}: {cortes[i - 1]:.2f} a {cortes[i]:.2f}")
        filas.append({"Variable": nodo, "Origen": col, "Niveles": " | ".join(partes)})
    return pd.DataFrame(filas)


def tabla_arcos(red: RedBayesiana) -> pd.DataFrame:
    """Flechas de la red como tabla (padre -> hijo)."""
    return pd.DataFrame(red.arcos_, columns=["Padre", "Hijo"])


def tabla_cpt(red: RedBayesiana, nodo: str) -> pd.DataFrame:
    """Tabla de probabilidad condicional de un nodo: P(nodo | sus padres).

    Cada fila es una combinación de estados de los padres y suma 1.
    """
    j = NODOS.index(nodo)
    ps = red.padres_[j]
    if ps:
        indice = pd.MultiIndex.from_product([red.estados_[NODOS[p]] for p in ps],
                                            names=[NODOS[p] for p in ps])
    else:
        indice = ["(sin padres)"]
    return pd.DataFrame(red.cpt_[j], index=indice, columns=red.estados_[nodo]).round(4)


# ---------------------------------------------------------------------------
# 5. Evaluación y comparación
# ---------------------------------------------------------------------------
def evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te, clase_critica="Maximum_Load"):
    """Métricas en entrenamiento y prueba (las mismas que los demás modelos)."""
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


def comparar_estructuras(X_tr, X_te, y_tr, y_te, n_bins: int = 3, max_indegree: int = 3,
                         alpha: float = 1.0, clase_critica="Maximum_Load"):
    """Entrena las tres estructuras y las compara en las mismas condiciones.

    Devuelve (tabla, redes, preds). La tabla incluye el número de flechas, de
    parámetros y el puntaje BIC en entrenamiento, además de las métricas.
    """
    configuraciones = {"Naive Bayes": "naive_bayes", "Experta": "experta",
                       "Aprendida (BIC)": "aprendida"}
    filas, redes, preds = [], {}, {}
    for nombre, est in configuraciones.items():
        red = RedBayesiana(est, n_bins=n_bins, max_indegree=max_indegree,
                           alpha=alpha).fit(X_tr, y_tr)
        metricas, pred = evaluar_modelo(red, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Estructura": nombre, "N_flechas": len(red.arcos_),
                      "N_parametros": red.n_parametros_, "BIC_train": round(red.bic_, 1),
                      **metricas})
        redes[nombre], preds[nombre] = red, pred
    return pd.DataFrame(filas), redes, preds


def comparar_n_bins(X_tr, X_te, y_tr, y_te, valores=(2, 3, 4, 5),
                    estructura="aprendida", clase_critica="Maximum_Load"):
    """Sensibilidad a la discretización: ¿cambia el resultado según los niveles?"""
    filas = []
    for k in valores:
        red = RedBayesiana(estructura, n_bins=k).fit(X_tr, y_tr)
        metricas, _ = evaluar_modelo(red, X_tr, X_te, y_tr, y_te, clase_critica)
        filas.append({"Niveles_por_variable": k, "N_flechas": len(red.arcos_),
                      "N_parametros": red.n_parametros_, **metricas})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 6. Gráficos
# ---------------------------------------------------------------------------
def _niveles_dag(padres):
    nivel = {}

    def n(j):
        if j not in nivel:
            nivel[j] = 0 if not padres[j] else 1 + max(n(p) for p in padres[j])
        return nivel[j]

    for j in padres:
        n(j)
    return nivel


def graficar_red(red: RedBayesiana, titulo: str = "", ax=None):
    """Dibuja la red: cada flecha va del padre (influye) al hijo (es influido)."""
    propia = ax is None
    if propia:
        fig, ax = plt.subplots(figsize=(7, 5.2))
    nivel = _niveles_dag(red.padres_)
    pos = {}
    for niv in sorted(set(nivel.values())):
        miembros = [j for j in range(len(NODOS)) if nivel[j] == niv]
        for k, j in enumerate(miembros):
            pos[j] = ((k - (len(miembros) - 1) / 2) * 2.0, -niv * 1.9)
    for h, ps in red.padres_.items():
        for p in ps:
            ax.annotate("", xy=pos[h], xytext=pos[p],
                        arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.3,
                                        shrinkA=24, shrinkB=24, mutation_scale=14))
    for j, (x, y) in pos.items():
        color = "#f5c26b" if j == 0 else "#bcd4ee"
        ax.text(x, y, _NOMBRES_VISIBLES[NODOS[j]], ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="gray"), zorder=3)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 1.1)
    ax.set_title(titulo)
    ax.axis("off")
    if propia:
        fig.tight_layout()
        return fig


def graficar_redes(redes: dict):
    """Dibuja varias redes lado a lado (la de color naranja es Load_Type)."""
    fig, axes = plt.subplots(1, len(redes), figsize=(6.2 * len(redes), 5.6))
    axes = np.atleast_1d(axes)
    for ax, (nombre, red) in zip(axes, redes.items()):
        graficar_red(red, f"{nombre} ({len(red.arcos_)} flechas)", ax=ax)
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