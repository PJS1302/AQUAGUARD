from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.decomposition import PCA

from backend.services.database_service import record_pca_run
from backend.utils.experiment_data import transform_unsupervised


def run_pca(frame: pd.DataFrame, features: list[str], n_components: int = 3, dataset_id: str | None = None) -> dict:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Selected PCA feature(s) are missing from the dataset.", "missing_features": missing})
    if not features or len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select one or more distinct numerical dataset features for PCA.")

    X, preprocessing = transform_unsupervised(frame, features, allow_categorical=False)
    if len(X) < 2:
        raise HTTPException(status_code=422, detail="PCA needs at least two rows to calculate sample variance.")
    if len(X) > 100_000:
        raise HTTPException(status_code=413, detail="PCA is limited to 100,000 rows per analysis.")
    if X.nunique(dropna=True).max() <= 1:
        raise HTTPException(status_code=422, detail="PCA requires at least one non-constant numerical feature.")
    max_components = min(X.shape[1], len(X))
    if not 1 <= int(n_components) <= max_components:
        raise HTTPException(status_code=422, detail=f"n_components must be between 1 and {max_components} for the selected numerical data.")

    model = PCA(n_components=int(n_components), random_state=42)
    scores = model.fit_transform(X.to_numpy(dtype=float))
    variance = model.explained_variance_ratio_.astype(float)
    loadings = model.components_.T
    transformed_features = preprocessing["transformed_feature_names"]
    result = {
        "original_feature_count": int(len(features)),
        "complete_rows": int(len(X)),
        "features": features,
        "component_count": int(n_components),
        "explained_variance_ratio": [float(value) for value in variance],
        "cumulative_explained_variance": [float(value) for value in np.cumsum(variance)],
        "loadings": [{"feature": str(feature), "components": [float(value) for value in values]} for feature, values in zip(transformed_features, loadings)],
        "scores": [{"pc1": float(row[0]), "pc2": float(row[1]) if len(row) > 1 else None, "index": int(idx)} for idx, row in zip(X.index[:300], scores[:300])],
        "feature_preprocessing": preprocessing,
        "note": "PCA uses selected numerical values. Missing numerical values are imputed with medians fitted on this unsupervised analysis set; no rows are discarded and the source dataset is unchanged.",
    }
    if dataset_id:
        record_pca_run(dataset_id, int(n_components), {"features": features, "explained_variance_ratio": result["explained_variance_ratio"], "cumulative_explained_variance": result["cumulative_explained_variance"]})
    return result
