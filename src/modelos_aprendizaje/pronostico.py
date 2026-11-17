"""Pronóstico del nivel de carga con consumo previo (Tarea 4 – DAEWOO Steel Co.).

Qué resuelve este módulo
------------------------
1. Construir las variables del pronóstico: con lo que se sabe en el instante t
   (consumo reciente, carga actual, mediciones eléctricas) y el calendario del
   instante que se pronostica (t + horizonte), predecir `Load_Type` en t + horizonte.
2. Partir los datos cronológicamente según el instante pronosticado.
3. Definir las líneas base (persistencia, calendario típico, clase mayoritaria).
4. Medir los modelos con métricas de pronóstico (acierto en cambios, anticipación
   de subidas a carga máxima) y compararlos.
5. Experimentos adicionales: ablación de grupos de variables y efecto del horizonte.

Uso típico (desde el notebook)
------------------------------
    from modelos_aprendizaje import preparacion as prep
    from modelos_aprendizaje import pronostico as pr

    df = prep.cargar_dataset_limpio()
    X_tr, X_te, y_tr, y_te, info = pr.preparar_pronostico(df, horizonte=4)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

ORDEN_CARGA = ["Light_Load", "Medium_Load", "Maximum_Load"]
CLASE_CRITICA = "Maximum_Load"
FECHA_CORTE = pd.Timestamp("2018-10-01")      # el instante pronosticado >= corte es de prueba
PASO_MIN = 15                                  # minutos entre lecturas

COL_LAGK = "Lagging_Current_Reactive.Power_kVarh"
COL_LEADK = "Leading_Current_Reactive_Power_kVarh"
COL_LAGPF = "Lagging_Current_Power_Factor"
COL_LEADPF = "Leading_Current_Power_Factor"

GRUPOS = {
    "Consumo previo": ["Consumo_t", "Consumo_t-15min", "Consumo_t-1h", "Consumo_t-24h",
                       "Variacion_1h", "Carga_actual"],
    "Mediciones eléctricas": [COL_LAGK, COL_LEADK, COL_LAGPF, COL_LEADPF],
    "Calendario del destino": ["Destino_WeekStatus_bin", "Destino_NSM_sin", "Destino_NSM_cos"],
}
_DESCRIPCION = {
    "Consumo_t": "Consumo actual (kWh)",
    "Consumo_t-15min": "Consumo de hace 15 minutos (kWh)",
    "Consumo_t-1h": "Consumo de hace 1 hora (kWh)",
    "Consumo_t-24h": "Consumo a la misma hora de ayer (kWh)",
    "Variacion_1h": "Consumo actual − consumo hace 1 h (tendencia)",
    "Carga_actual": "Nivel de carga actual (0 liviana, 1 media, 2 máxima)",
    COL_LAGK: "Energía reactiva inductiva actual",
    COL_LEADK: "Energía reactiva capacitiva actual",
    COL_LAGPF: "Factor de potencia inductivo actual",
    COL_LEADPF: "Factor de potencia capacitivo actual",
    "Destino_WeekStatus_bin": "Tipo de día del instante pronosticado (0 hábil, 1 fin de semana)",
    "Destino_NSM_sin": "Seno de la hora del instante pronosticado",
    "Destino_NSM_cos": "Coseno de la hora del instante pronosticado",
}


# ---------------------------------------------------------------------------
# 1. Construcción de variables y partición
# ---------------------------------------------------------------------------
def _construir(df: pd.DataFrame, horizonte: int):
    """Arma X, y y el instante pronosticado a partir del dataset limpio.

    Todo se calcula con desplazamientos *en el tiempo* sobre una grilla completa de
    15 minutos, de modo que si faltara una lectura el rezago queda vacío y la fila se
    descarta (no se usa una lectura equivocada).
    """
    d = df.set_index("date").sort_index()
    grilla = pd.date_range(d.index.min(), d.index.max(), freq=f"{PASO_MIN}min")
    d = d.reindex(grilla)
    carga = d["Load_Type"].map({c: i for i, c in enumerate(ORDEN_CARGA)})
    uso = d["Usage_kWh"]

    destino = d.index + pd.Timedelta(minutes=PASO_MIN * horizonte)
    seg = (destino.hour * 3600 + destino.minute * 60).to_numpy()
    ang = 2 * np.pi * seg / 86400.0

    X = pd.DataFrame({
        "Consumo_t": uso,
        "Consumo_t-15min": uso.shift(1),
        "Consumo_t-1h": uso.shift(4),
        "Consumo_t-24h": uso.shift(96),
        "Carga_actual": carga,
        COL_LAGK: d[COL_LAGK], COL_LEADK: d[COL_LEADK],
        COL_LAGPF: d[COL_LAGPF], COL_LEADPF: d[COL_LEADPF],
        "Destino_WeekStatus_bin": (destino.dayofweek >= 5).astype(int),
        "Destino_NSM_sin": np.sin(ang), "Destino_NSM_cos": np.cos(ang),
    }, index=d.index)
    X["Variacion_1h"] = X["Consumo_t"] - X["Consumo_t-1h"]
    X = X[[c for g in GRUPOS.values() for c in g]]

    y = d["Load_Type"].shift(-horizonte)
    y.name = "Carga_futura"
    fecha_obj = pd.Series(destino, index=d.index, name="fecha_objetivo")

    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok], fecha_obj[ok]


def preparar_pronostico(df: pd.DataFrame, horizonte: int = 4, fecha_corte=FECHA_CORTE):
    """Construye las variables y parte en entrenamiento y prueba por fecha pronosticada.

    Devuelve (X_tr, X_te, y_tr, y_te, info). El índice de X es el instante de origen t.
    `info` trae: filas descartadas, fechas pronosticadas de cada grupo (Series con el
    mismo índice que X) y los grupos de variables.
    """
    X, y, fecha_obj = _construir(df, horizonte)
    es_te = (fecha_obj >= pd.Timestamp(fecha_corte)).to_numpy()
    info = {
        "horizonte": horizonte,
        "n_filas_descartadas": len(df) - len(X),
        "fecha_objetivo_tr": fecha_obj[~es_te],
        "fecha_objetivo_te": fecha_obj[es_te],
        "grupos": {k: list(v) for k, v in GRUPOS.items()},
    }
    return X[~es_te], X[es_te], y[~es_te], y[es_te], info


def tabla_variables(X) -> pd.DataFrame:
    """Lista de variables del pronóstico, su grupo y qué representan."""
    filas = [{"Variable": c, "Grupo": g, "Qué representa": _DESCRIPCION[c]}
             for g, cols in GRUPOS.items() for c in cols if c in X.columns]
    return pd.DataFrame(filas)


def carga_actual_texto(X) -> pd.Series:
    """Nivel de carga actual como texto (Light_Load, ...)."""
    return X["Carga_actual"].astype(int).map(dict(enumerate(ORDEN_CARGA)))


def hora_destino(X) -> pd.Series:
    """Hora (0 a 24) del instante pronosticado, a partir de su seno y coseno."""
    ang = np.arctan2(X["Destino_NSM_sin"].to_numpy(), X["Destino_NSM_cos"].to_numpy()) % (2 * np.pi)
    return pd.Series(ang / (2 * np.pi) * 24.0, index=X.index, name="hora_destino")


# ---------------------------------------------------------------------------
# 2. Qué tanto cambia la carga
# ---------------------------------------------------------------------------
def tabla_transiciones(X, y) -> pd.DataFrame:
    """Matriz de transición (%): filas = carga actual, columnas = carga dentro del horizonte."""
    t = pd.crosstab(carga_actual_texto(X).rename("Carga actual"), y.rename("Carga futura"),
                    normalize="index") * 100
    return t.reindex(index=ORDEN_CARGA, columns=ORDEN_CARGA).round(2)


def resumen_cambios(X, y) -> dict:
    """Cuántos casos son cambios y cuántos son subidas a carga máxima."""
    actual = carga_actual_texto(X)
    cambia = actual.to_numpy() != y.to_numpy()
    sube = cambia & (y.to_numpy() == CLASE_CRITICA)
    return {
        "Casos": int(len(y)),
        "Casos en que la carga cambia": int(cambia.sum()),
        "% de cambios": round(100 * cambia.mean(), 2),
        "Subidas a carga máxima": int(sube.sum()),
        "% de subidas a máxima": round(100 * sube.mean(), 2),
    }


# ---------------------------------------------------------------------------
# 3. Líneas base
# ---------------------------------------------------------------------------
class Persistencia:
    """'Dentro del horizonte la carga será igual a la actual.'"""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return carga_actual_texto(X).to_numpy()


class CalendarioTipico:
    """'La carga más frecuente a esa hora y tipo de día' (solo mira el calendario)."""

    def _clave(self, X):
        slot = np.round(hora_destino(X).to_numpy() * 4).astype(int) % 96
        return pd.MultiIndex.from_arrays([slot, X["Destino_WeekStatus_bin"].to_numpy()])

    def fit(self, X, y):
        clave = self._clave(X)
        s = pd.Series(np.asarray(y), index=clave)
        self.global_ = s.mode().iloc[0]
        self.tabla_ = s.groupby(level=[0, 1]).agg(lambda v: v.mode().iloc[0])
        return self

    def predict(self, X):
        return self.tabla_.reindex(self._clave(X)).fillna(self.global_).to_numpy()


class ClaseMayoritaria:
    """'Siempre la clase más frecuente del entrenamiento.'"""

    def fit(self, X, y):
        self.clase_ = pd.Series(np.asarray(y)).mode().iloc[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.clase_, dtype=object)


def lineas_base(X_tr, y_tr) -> dict:
    """Entrena las tres líneas base. La clave de la persistencia la usa el notebook."""
    return {
        "Persistencia (carga actual)": Persistencia().fit(X_tr, y_tr),
        "Calendario típico": CalendarioTipico().fit(X_tr, y_tr),
        "Clase mayoritaria": ClaseMayoritaria().fit(X_tr, y_tr),
    }


# ---------------------------------------------------------------------------
# 4. Métricas de pronóstico y comparación
# ---------------------------------------------------------------------------
def metricas_desde_pred(pred, X, y, clase_critica=CLASE_CRITICA) -> dict:
    """Métricas de pronóstico a partir de las predicciones (X debe traer Carga_actual)."""
    pred, y = np.asarray(pred, dtype=object), np.asarray(y, dtype=object)
    actual = carga_actual_texto(X).to_numpy()
    cambia = actual != y
    sube = cambia & (y == clase_critica)
    return {
        "Accuracy_test": accuracy_score(y, pred),
        "F1_macro_test": f1_score(y, pred, average="macro"),
        f"Recall_{clase_critica}": float(recall_score(y, pred, labels=[clase_critica],
                                                       average=None)[0]),
        f"Precision_{clase_critica}": float(precision_score(
            y, pred, labels=[clase_critica], average=None, zero_division=0)[0]),
        "Acierto_cambios": float((pred[cambia] == y[cambia]).mean()) if cambia.any() else np.nan,
        "Recall_subidas_Max": float((pred[sube] == y[sube]).mean()) if sube.any() else np.nan,
    }


def comparar_modelos(modelos: dict, X_tr, X_te, y_tr, y_te):
    """Tabla con las mismas métricas para modelos y líneas base. Devuelve (tabla, preds)."""
    filas, preds = {}, {}
    for nombre, m in modelos.items():
        pred_te = np.asarray(m.predict(X_te), dtype=object)
        met = metricas_desde_pred(pred_te, X_te, y_te)
        acc_tr = accuracy_score(y_tr, np.asarray(m.predict(X_tr), dtype=object))
        filas[nombre] = {"Accuracy_train": acc_tr, **met}
        preds[nombre] = pred_te
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "Modelo"
    return tabla, preds


# ---------------------------------------------------------------------------
# 5. Experimentos adicionales
# ---------------------------------------------------------------------------
_PARAMS_RF = dict(n_estimators=200, max_depth=10, min_samples_leaf=20,
                  max_features="sqrt")


def _rf(params_rf=None):
    """Random Forest con los hiperparámetros dados (o unos por defecto)."""
    return RandomForestClassifier(**{**_PARAMS_RF, **(params_rf or {})},
                                  random_state=42, n_jobs=-1)


def ablacion_grupos(X_tr, X_te, y_tr, y_te, grupos: dict | None = None,
                    params_rf: dict | None = None) -> pd.DataFrame:
    """Mismo Random Forest con distintos grupos de variables (experimento de ablación).

    `params_rf`: hiperparámetros del bosque (conviene pasar los elegidos en la sección 6).
    """
    grupos = grupos or GRUPOS
    cal, ele, con = (grupos["Calendario del destino"], grupos["Mediciones eléctricas"],
                     grupos["Consumo previo"])
    conjuntos = {
        "Solo calendario del destino": cal,
        "Calendario + mediciones eléctricas": cal + ele,
        "Calendario + consumo previo": cal + con,
        "Todas las variables": cal + ele + con,
    }
    filas = []
    for nombre, cols in conjuntos.items():
        m = _rf(params_rf).fit(X_tr[cols], y_tr)
        met = metricas_desde_pred(m.predict(X_te[cols]), X_te, y_te)
        filas.append({"Variables": nombre, "N_variables": len(cols), **met})
    return pd.DataFrame(filas).set_index("Variables")


def curva_horizonte(df, horizontes=(1, 2, 4, 8, 16),
                    params_rf: dict | None = None) -> pd.DataFrame:
    """Persistencia y Random Forest según qué tan lejos se pronostica."""
    filas = []
    for h in horizontes:
        X_tr, X_te, y_tr, y_te, _ = preparar_pronostico(df, horizonte=h)
        pers = Persistencia()
        rf = _rf(params_rf).fit(X_tr, y_tr)
        for nombre, pred in (("Persistencia", pers.predict(X_te)),
                             ("Random Forest", rf.predict(X_te))):
            filas.append({"Horizonte_pasos": h, "Horizonte_min": h * PASO_MIN,
                          "Modelo": nombre,
                          **metricas_desde_pred(pred, X_te, y_te)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 6. Gráficos
# ---------------------------------------------------------------------------
def graficar_comparacion(tabla: pd.DataFrame):
    """Barras horizontales: F1 macro, recall de carga máxima y recall de subidas a máxima."""
    cols = ["F1_macro_test", f"Recall_{CLASE_CRITICA}", "Recall_subidas_Max"]
    etiquetas = ["F1 macro", "Recall carga máxima", "Recall subidas a máxima"]
    datos = tabla[cols].astype(float).fillna(0)
    ax = datos.plot.barh(figsize=(9, 0.9 * len(datos) + 2), width=0.8)
    ax.legend(etiquetas, loc="lower right")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Valor en el conjunto de prueba")
    ax.set_title("Comparación de modelos y líneas base")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return ax.figure


def graficar_horizonte(curva: pd.DataFrame, metrica: str = "Accuracy_test"):
    """Una línea por modelo: métrica según el horizonte (en minutos)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for nombre, g in curva.groupby("Modelo"):
        ax.plot(g["Horizonte_min"], g[metrica], marker="o", label=nombre)
    ax.set_xlabel("Horizonte de pronóstico (minutos)")
    ax.set_ylabel(metrica)
    ax.set_title(f"{metrica} según el horizonte")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def graficar_matrices_confusion(preds: dict, y_te, orden_clases=ORDEN_CARGA):
    """Matrices de confusión (test) de varios modelos, lado a lado."""
    fig, axes = plt.subplots(1, len(preds), figsize=(5.0 * len(preds), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (nombre, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_te, pred, labels=orden_clases)
        ConfusionMatrixDisplay(cm, display_labels=orden_clases).plot(
            ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
        ax.set_title(nombre, fontsize=10)
    fig.tight_layout()
    return fig
