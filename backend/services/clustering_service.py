from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from backend.services.database_service import record_clustering_run
from backend.utils.experiment_data import require_complete_processed_rows, require_finite_numeric


def prepare_cluster_data(frame: pd.DataFrame, features: list[str], n_clusters: int) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Selected processed clustering feature(s) are missing.", "missing_features": missing})
    if not features or len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select one or more distinct processed clustering features.")

    X_df = require_complete_processed_rows(frame, features, task="Clustering", minimum=1)
    require_finite_numeric(X_df, features, task="Clustering")
    if len(X_df) < max(4, int(n_clusters) + 1):
        raise HTTPException(status_code=422, detail="Not enough complete processed samples for the selected cluster count.")
    if len(X_df) > 5_000:
        raise HTTPException(status_code=413, detail="Clustering (including pairwise internal metrics and hierarchical linkage) is limited to 5,000 processed rows per analysis.")
    # The configured user's preprocessed values are used as-is; no scaling is added.
    values = X_df.to_numpy(dtype=float)
    return X_df, values


def _metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None, "message": "Internal clustering metrics require at least 2 clusters and fewer clusters than samples."}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def _sizes(labels: np.ndarray) -> dict:
    keys, counts = np.unique(labels, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(keys, counts)}


def _points(X: np.ndarray, labels: np.ndarray) -> list[dict]:
    y_values = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
    return [{"x": float(x), "y": float(y), "cluster": int(label), "index": int(i)} for i, (x, y, label) in enumerate(zip(X[:, 0], y_values, labels))]


def _dendrogram_data(X: np.ndarray) -> dict | None:
    if len(X) < 3:
        return None
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        sample = X[: min(len(X), 300)]
        tree = linkage(sample, method="ward")
        plotted = dendrogram(tree, truncate_mode="lastp", p=12, no_plot=True)
        return {"icoord": plotted.get("icoord", []), "dcoord": plotted.get("dcoord", []), "leaf_labels": plotted.get("ivl", []), "sample_rows": int(len(sample)), "truncated": len(X) > len(sample)}
    except Exception:
        return None


def _make_result(algorithm: str, X_df: pd.DataFrame, values: np.ndarray, labels: np.ndarray, n_clusters: int, dataset_id: str, *, centers=None, dendrogram=None) -> dict:
    metrics = _metrics(values, labels)
    sizes = _sizes(labels)
    result = {
        "algorithm": algorithm,
        "dataset_id": dataset_id,
        "features": [str(column) for column in X_df.columns],
        "cluster_count": int(n_clusters),
        "cluster_sizes": sizes,
        "metrics": metrics,
        "points": _points(values, labels)[:1000],
        "labels": [int(value) for value in labels[:2000]],
        "complete_rows": int(len(X_df)),
        "cluster_assignments_truncated": len(X_df) > 2000,
        "scaling_applied_by_aquaguard": False,
    }
    if centers is not None:
        result["cluster_centers"] = [{str(feature): float(value) for feature, value in zip(X_df.columns, row)} for row in centers]
    if dendrogram is not None:
        result["dendrogram"] = dendrogram
    record_clustering_run(algorithm, dataset_id, n_clusters, metrics)
    return result


def run_kmeans(frame: pd.DataFrame, dataset_id: str, features: list[str], n_clusters: int = 3) -> dict:
    X_df, values = prepare_cluster_data(frame, features, n_clusters)
    model = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
    labels = model.fit_predict(values)
    return _make_result("K-Means", X_df, values, labels, n_clusters, dataset_id, centers=model.cluster_centers_)


def run_hierarchical(frame: pd.DataFrame, dataset_id: str, features: list[str], n_clusters: int = 3) -> dict:
    X_df, values = prepare_cluster_data(frame, features, n_clusters)
    model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage="ward")
    labels = model.fit_predict(values)
    return _make_result("Hierarchical / Agglomerative", X_df, values, labels, n_clusters, dataset_id, dendrogram=_dendrogram_data(values))


def compare_clustering(frame: pd.DataFrame, dataset_id: str, features: list[str], n_clusters: int = 3) -> dict:
    X_df, values = prepare_cluster_data(frame, features, n_clusters)
    km = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
    km_labels = km.fit_predict(values)
    ag = AgglomerativeClustering(n_clusters=int(n_clusters), linkage="ward")
    ag_labels = ag.fit_predict(values)
    km_result = _make_result("K-Means", X_df, values, km_labels, n_clusters, dataset_id, centers=km.cluster_centers_)
    ag_result = _make_result("Hierarchical / Agglomerative", X_df, values, ag_labels, n_clusters, dataset_id, dendrogram=_dendrogram_data(values))
    km_score = km_result["metrics"].get("silhouette")
    ag_score = ag_result["metrics"].get("silhouette")
    if km_score is None or ag_score is None:
        comment = "A silhouette-based comparison is unavailable for this dataset and cluster assignment. Inspect the cluster sizes and feature scatter before drawing a conclusion."
    elif abs(km_score - ag_score) < 1e-9:
        comment = "Both algorithms produced the same silhouette score at the displayed precision; no separation winner is identified by this metric."
    elif km_score > ag_score:
        comment = f"K-Means has the higher silhouette score ({km_score:.4f} vs {ag_score:.4f}) on these user-preprocessed values, indicating better average separation by this metric."
    else:
        comment = f"Hierarchical clustering has the higher silhouette score ({ag_score:.4f} vs {km_score:.4f}) on these user-preprocessed values, indicating better average separation by this metric."
    return {"dataset_id": dataset_id, "features": features, "cluster_count": int(n_clusters), "kmeans": km_result, "hierarchical": ag_result, "automatic_comment": comment, "interpretation_note": "Clusters are exploratory groups, not Safe/Unsafe labels. Clustering used the user's processed numeric values without additional scaling."}
