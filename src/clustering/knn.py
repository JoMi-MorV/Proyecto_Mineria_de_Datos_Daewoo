"""KNN con K = 3, 4, 5 y 6: entrenamiento, métricas y gráficos."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, ConfusionMatrixDisplay)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KS = (3, 4, 5, 6)


def _pipe(modelo):
    # El scaler vive dentro del Pipeline: se ajusta solo con el train.
    return Pipeline([("scaler", StandardScaler()), ("knn", modelo)])


# ---------------------------------------------------------------- métricas
def mape(y, p, umbral: float = 1.0):
    """MAPE solo sobre y > umbral (kWh). Con consumos ~0 el MAPE se infla."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > umbral
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    r = np.where(den == 0, 0.0, 2 * np.abs(y - p) / np.where(den == 0, 1, den))
    return float(np.mean(r) * 100)


def metricas_regresion(y, p, umbral: float = 1.0) -> dict:
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            f"MAPE(y>{umbral:g})": mape(y, p, umbral),
            "sMAPE": smape(y, p)}


# ------------------------------------------------------------ clasificación
def contar_empates(pipe, X_tr, y_tr, X_te) -> float:
    """% de casos de prueba con empate de votos entre vecinos (relevante con K par)."""
    knn = pipe.named_steps["knn"]
    idx = knn.kneighbors(pipe.named_steps["scaler"].transform(X_te), return_distance=False)
    etiquetas = np.asarray(y_tr)[idx]
    conteos = np.stack([(etiquetas == c).sum(1) for c in knn.classes_], axis=1)
    return float(((conteos == conteos.max(1, keepdims=True)).sum(1) > 1).mean() * 100)


def evaluar_clasificacion(X_tr, X_te, y_tr, y_te, ks=KS, weights="uniform"):
    filas, modelos, preds = [], {}, {}
    for k in ks:
        pipe = _pipe(KNeighborsClassifier(n_neighbors=k, weights=weights)).fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=TimeSeriesSplit(5), scoring="f1_macro").mean()
        filas.append({"K": k, "weights": weights,
                      "Accuracy": accuracy_score(y_te, p),
                      "F1_macro": f1_score(y_te, p, average="macro"),
                      "F1_macro_CV(train)": cv,
                      "%Empates": contar_empates(pipe, X_tr, y_tr, X_te)})
        modelos[k], preds[k] = pipe, p
    return pd.DataFrame(filas), modelos, preds


def graficar_matrices_confusion(preds, y_te, labels, titulo=""):
    fig, axs = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4))
    for ax, (k, p) in zip(np.atleast_1d(axs), preds.items()):
        ConfusionMatrixDisplay.from_predictions(y_te, p, labels=labels, ax=ax,
                                                colorbar=False, cmap="Blues",
                                                xticks_rotation=30)
        ax.set_title(f"K = {k}")
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------- regresión
def linea_base_persistencia(X_te, y_te, umbral: float = 1.0) -> dict:
    """Referencia ingenua: 'el consumo futuro será igual al actual'."""
    return metricas_regresion(y_te, X_te["Usage_actual"], umbral)


def evaluar_regresion(X_tr, X_te, y_tr, y_te, ks=KS, weights="uniform", umbral=1.0):
    filas, modelos, preds = [], {}, {}
    for k in ks:
        pipe = _pipe(KNeighborsRegressor(n_neighbors=k, weights=weights)).fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        cv = -cross_val_score(pipe, X_tr, y_tr, cv=TimeSeriesSplit(5),
                              scoring="neg_mean_absolute_error").mean()
        filas.append({"K": k, "weights": weights, **metricas_regresion(y_te, p, umbral),
                      "MAE_CV(train)": cv})
        modelos[k], preds[k] = pipe, p
    return pd.DataFrame(filas), modelos, preds


# ---------------------------------------------------------------- utilidades
def graficar_metricas_vs_k(tabla, metricas, titulo=""):
    fig, axs = plt.subplots(1, len(metricas), figsize=(4.5 * len(metricas), 3.8))
    for ax, m in zip(np.atleast_1d(axs), metricas):
        for w, g in tabla.groupby("weights"):
            ax.plot(g["K"], g[m], marker="o", label=w)
        ax.set_xticks(sorted(tabla["K"].unique()))
        ax.set_xlabel("K"); ax.set_title(m); ax.grid(alpha=.3); ax.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()


def elegir_mejor(tabla, columna, mayor_es_mejor=True):
    """Fila de la tabla con el mejor valor de `columna` (usar columna CV para no mirar el test)."""
    i = tabla[columna].idxmax() if mayor_es_mejor else tabla[columna].idxmin()
    return tabla.loc[i]


def graficar_real_vs_pred(fechas, y_real, y_pred, k, desde, hasta, titulo=""):
    f = pd.Series(np.asarray(fechas))
    m = ((f >= desde) & (f < hasta)).to_numpy()
    plt.figure(figsize=(11, 3.6))
    plt.plot(f[m], np.asarray(y_real)[m], label="Real", lw=1.3)
    plt.plot(f[m], np.asarray(y_pred)[m], label=f"KNN K={k}", lw=1.3, alpha=.85)
    plt.ylabel("Usage_kWh"); plt.title(titulo); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()
