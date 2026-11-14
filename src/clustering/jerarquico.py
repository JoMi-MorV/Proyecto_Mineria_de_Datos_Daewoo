import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score


def tomar_muestra(df, n=4000, semilla=42):
    """Muestra aleatoria reproducible (la matriz de distancias completa
    de 35.040 registros ocuparía ~4,9 GB)."""
    return df.sample(n=n, random_state=semilla)


def validar_muestra(df, muestra, cols_num, col_cat="Load_Type"):
    """Compara medias y proporciones de la categoría entre dataset completo y muestra."""
    medias = pd.DataFrame({
        "completo": df[cols_num].mean(),
        "muestra": muestra[cols_num].mean(),
    })
    medias["dif_%"] = 100 * (medias["muestra"] - medias["completo"]) / medias["completo"]
    props = pd.DataFrame({
        "completo_%": df[col_cat].value_counts(normalize=True) * 100,
        "muestra_%": muestra[col_cat].value_counts(normalize=True) * 100,
    }).round(2)
    return medias.round(3), props


def ejecutar_jerarquico(X, metodo, k, metrica="euclidean"):
    """Una ejecución: linkage, cofenético, corte en k clusters y silueta."""
    d = pdist(X, metric=metrica)
    Z = linkage(d, method=metodo)
    coef_cofenetico, _ = cophenet(Z, d)
    etiquetas = fcluster(Z, t=k, criterion="maxclust")
    sil = silhouette_score(X, etiquetas, metric=metrica)
    return {"metodo": metodo, "Z": Z, "etiquetas": etiquetas,
            "cofenetico": coef_cofenetico, "silueta": sil}


def graficar_dendrograma(Z, metodo, k, p=30):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=p, show_leaf_counts=True,
               leaf_rotation=90, ax=ax)
    # línea de corte aproximada para k clusters
    altura = (Z[-(k - 1), 2] + Z[-k, 2]) / 2
    ax.axhline(altura, color="red", ls="--", label=f"Corte en {k} clusters")
    ax.set_title(f"Dendrograma ({metodo}, euclidiana), últimos {p} nodos")
    ax.set_xlabel("Cluster (n.º de registros)")
    ax.set_ylabel("Distancia de enlace")
    ax.legend()
    plt.tight_layout()
    plt.show()


def perfil_clusters(muestra, etiquetas, cols_perfil):
    """Tamaño y medias de las variables originales (sin escalar) por cluster."""
    tmp = muestra[cols_perfil].copy()
    tmp["cluster"] = etiquetas
    perfil = tmp.groupby("cluster").mean().round(2)
    perfil.insert(0, "n", tmp["cluster"].value_counts().sort_index())
    perfil.insert(1, "%", (perfil["n"] / len(tmp) * 100).round(2))
    return perfil


def cruzar_con_categoria(muestra, etiquetas, col_cat="Load_Type"):
    """% de cada categoría dentro de cada cluster (filas suman 100)."""
    return (pd.crosstab(etiquetas, muestra[col_cat].values, normalize="index") * 100).round(1)


def tabla_comparativa(resultados):
    return pd.DataFrame([
        {"Método": r["metodo"],
         "Coef. cofenético": round(r["cofenetico"], 4),
         "Silueta": round(r["silueta"], 4),
         "Tamaños": sorted(np.bincount(r["etiquetas"])[1:], reverse=True)}
        for r in resultados.values()
    ])


def matriz_ari(resultados):
    nombres = list(resultados)
    m = pd.DataFrame(index=nombres, columns=nombres, dtype=float)
    for a in nombres:
        for b in nombres:
            m.loc[a, b] = adjusted_rand_score(resultados[a]["etiquetas"],
                                              resultados[b]["etiquetas"])
    return m.round(3)