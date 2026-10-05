from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.decomposition import PCA

from backend.services.database_service import record_pca_run
from backend.utils.experiment_data import require_complete_processed_rows, require_finite_numeric


def run_pca(frame: pd.DataFrame, features: list[str], n_components: int = 3, dataset_id: str | None = None) -> dict:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Selected processed PCA feature(s) are missing.", "missing_features": missing})
    if not features or len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select one or more distinct processed PCA features.")

    X = require_complete_processed_rows(frame, features, task="PCA", minimum=1)
    require_finite_numeric(X, features, task="PCA")
    if len(X) < 3:
        raise HTTPException(status_code=422, detail="PCA requires at least 3 complete processed rows.")
    if len(X) > 100_000:
        raise HTTPException(status_code=413, detail="PCA is limited to 100,000 processed rows per analysis.")
    if X.nunique(dropna=True).max() <= 1:
        raise HTTPException(status_code=422, detail="PCA requires at least one non-constant processed numerical feature.")
    max_components = min(len(features), len(X))
    if not 1 <= int(n_components) <= max_components:
        raise HTTPException(status_code=422, detail=f"n_components must be between 1 and {max_components} for this processed dataset.")

    # PCA is the requested analysis. No additional scaling or cleaning is applied.
    model = PCA(n_components=int(n_components), random_state=42)
    scores = model.fit_transform(X.to_numpy(dtype=float))
    variance = model.explained_variance_ratio_.astype(float)
    loadings = model.components_.T
    result = {
        "original_feature_count": int(len(features)),
        "complete_rows": int(len(X)),
        "features": features,
        "component_count": int(n_components),
        "explained_variance_ratio": [float(value) for value in variance],
        "cumulative_explained_variance": [float(value) for value in np.cumsum(variance)],
        "loadings": [{"feature": str(feature), "components": [float(value) for value in values]} for feature, values in zip(features, loadings)],
        "scores": [{"pc1": float(row[0]), "pc2": float(row[1]) if len(row) > 1 else 0.0, "index": int(idx)} for idx, row in zip(X.index[:300], scores[:300])],
        "note": "PCA is an exploratory reduction/visualization over the user's processed numeric values. AquaGuard adds no scaler; PCA is not inserted into production unless the supplied production artifact includes it.",
    }
    if dataset_id:
        record_pca_run(dataset_id, int(n_components), {"features": features, "explained_variance_ratio": result["explained_variance_ratio"], "cumulative_explained_variance": result["cumulative_explained_variance"]})
    return result
