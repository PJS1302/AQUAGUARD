from __future__ import annotations
import math
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from backend.services.database_service import record_regression_run
from backend.utils.experiment_data import require_complete_processed_rows, require_finite_numeric


def prepare_regression_data(frame: pd.DataFrame, target: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series, int]:
    if target not in frame.columns:
        raise HTTPException(status_code=422, detail=f"Target column '{target}' was not found in the user's processed dataset.")
    if len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select each processed feature only once.")
    missing_columns = [c for c in features if c not in frame.columns]
    if missing_columns:
        raise HTTPException(status_code=422, detail={"message": "Selected processed regression feature(s) are missing.", "missing_features": missing_columns})
    if target in features:
        raise HTTPException(status_code=422, detail="The target cannot also be an input feature.")
    if len(features) < 1:
        raise HTTPException(status_code=422, detail="Select at least one processed input feature.")

    columns = features + [target]
    data = require_complete_processed_rows(frame, columns, task="Regression")
    require_finite_numeric(data, columns, task="Regression")
    y = data[target].astype(float)
    if y.nunique() < 3:
        raise HTTPException(status_code=422, detail="The selected target has fewer than 3 distinct values and appears categorical. Regression requires a continuous numerical target.")
    return data[features].astype(float), y, 0


def _metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _cross_validation(estimator, X: pd.DataFrame, y: pd.Series, folds: int) -> dict:
    usable_folds = min(int(folds), len(X) // 2)
    if usable_folds < 2:
        return {"folds": 0, "fold_scores": [], "mean_score": None, "std_dev": None, "metric": "R²", "message": "Not enough complete rows for K-Fold R² evaluation."}
    splitter = KFold(n_splits=usable_folds, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(estimator, X, y, cv=splitter, scoring="r2", error_score="raise")
    except Exception as exc:
        return {"folds": usable_folds, "fold_scores": [], "mean_score": None, "std_dev": None, "metric": "R²", "message": f"Cross-validation could not be calculated: {exc}"}
    return {"folds": usable_folds, "fold_scores": [float(s) for s in scores], "mean_score": float(np.mean(scores)), "std_dev": float(np.std(scores, ddof=0)), "metric": "R²"}


def _lwlr_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, bandwidth: float) -> np.ndarray:
    # Distances use the exact user-preprocessed values; AquaGuard adds no scaler.
    train = np.asarray(X_train, dtype=float)
    queries = np.asarray(X_query, dtype=float)
    design = np.column_stack([np.ones(len(train)), train])
    out = []
    for query in queries:
        query_design = np.r_[1.0, query]
        distance_sq = np.sum((train - query) ** 2, axis=1)
        weights = np.exp(-distance_sq / (2.0 * float(bandwidth) ** 2))
        sqrt_w = np.sqrt(np.maximum(weights, 1e-12))
        weighted_design = design * sqrt_w[:, None]
        weighted_y = y_train * sqrt_w
        coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        out.append(float(query_design @ coefficients))
    return np.asarray(out, dtype=float)


def _lwlr_cv(X: pd.DataFrame, y: pd.Series, requested_folds: int, bandwidth: float) -> dict:
    folds = min(int(requested_folds), len(X) // 2)
    if folds < 2:
        return {"folds": 0, "fold_scores": [], "mean_score": None, "std_dev": None, "metric": "R²", "message": "Not enough complete rows for K-Fold R² evaluation."}
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in splitter.split(X):
        pred = _lwlr_predict(X.iloc[train_idx].to_numpy(float), y.iloc[train_idx].to_numpy(float), X.iloc[test_idx].to_numpy(float), bandwidth)
        scores.append(float(r2_score(y.iloc[test_idx], pred)))
    return {"folds": folds, "fold_scores": scores, "mean_score": float(np.mean(scores)), "std_dev": float(np.std(scores, ddof=0)), "metric": "R²"}


def run_regression(frame: pd.DataFrame, *, dataset_id: str, algorithm: str, target: str, features: list[str], degree: int = 2, folds: int = 5, bandwidth: float = 0.8, query: dict | None = None) -> dict:
    X, y, _ = prepare_regression_data(frame, target, features)
    if algorithm == "linear" and len(features) != 1:
        raise HTTPException(status_code=422, detail="Linear Regression uses exactly one input feature. Use Multiple Linear Regression for two or more features.")
    if algorithm == "multiple" and len(features) < 2:
        raise HTTPException(status_code=422, detail="Multiple Linear Regression requires at least two input features.")
    if algorithm not in {"linear", "multiple", "polynomial", "lwlr"}:
        raise HTTPException(status_code=400, detail="Unknown regression algorithm.")
    if algorithm == "polynomial" and not (2 <= int(degree) <= 5):
        raise HTTPException(status_code=422, detail="Polynomial degree must be between 2 and 5.")
    if algorithm == "polynomial" and math.comb(len(features) + int(degree), int(degree)) - 1 > 5_000:
        raise HTTPException(status_code=422, detail="Polynomial expansion would exceed 5,000 terms. Reduce feature count or degree.")
    if algorithm == "lwlr" and len(X) > 20_000:
        raise HTTPException(status_code=413, detail="LWLR is quadratic in the number of samples and is limited to 20,000 complete rows per run.")
    if algorithm == "lwlr" and not (0 < float(bandwidth) <= 20):
        raise HTTPException(status_code=422, detail="Bandwidth must be greater than zero and no more than 20.")
    n_test = max(2, int(math.ceil(len(X) * 0.2)))
    if len(X) - n_test < 2:
        raise HTTPException(status_code=422, detail="Not enough rows to create a train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)

    if algorithm == "lwlr":
        predictions = _lwlr_predict(X_train.to_numpy(float), y_train.to_numpy(float), X_test.to_numpy(float), bandwidth)
        metrics = _metrics(y_test, predictions)
        cv = _lwlr_cv(X, y, folds, bandwidth)
        if query:
            missing_query = [f for f in features if f not in query]
            if missing_query:
                raise HTTPException(status_code=422, detail={"message": "LWLR query point is missing selected feature(s).", "missing_features": missing_query})
            query_frame = pd.DataFrame([{f: float(query[f]) for f in features}], columns=features)
            query_prediction = float(_lwlr_predict(X.to_numpy(float), y.to_numpy(float), query_frame.to_numpy(float), bandwidth)[0])
            query_point = {k: float(v) for k, v in query_frame.iloc[0].items()}
        else:
            query_point = {k: float(v) for k, v in X_test.iloc[0].items()}
            query_prediction = float(predictions[0])
        result = {
            "algorithm": "Locally Weighted Linear Regression (LWLR)",
            "target": target, "features": features, "bandwidth": float(bandwidth),
            "query_point": query_point, "query_prediction": query_prediction,
            "actual": [float(v) for v in y_test.to_numpy()[:200]],
            "predicted": [float(v) for v in predictions[:200]],
            "metrics": metrics, "cross_validation": cv, "complete_rows": int(len(X)),
        }
    else:
        if algorithm == "polynomial":
            estimator = make_pipeline(PolynomialFeatures(degree=int(degree), include_bias=False), LinearRegression())
        else:
            estimator = LinearRegression()
        try:
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Regression could not be fitted: {exc}") from exc
        metrics = _metrics(y_test, predictions)
        cv = _cross_validation(estimator, X, y, folds)
        if algorithm == "polynomial":
            names = estimator.named_steps["polynomialfeatures"].get_feature_names_out(features).tolist()
            fitted = estimator.named_steps["linearregression"]
            coefficients = {str(k): float(v) for k, v in zip(names, fitted.coef_)}
            intercept = float(fitted.intercept_)
        else:
            coefficients = {str(k): float(v) for k, v in zip(features, estimator.coef_)}
            intercept = float(estimator.intercept_)
        result = {
            "algorithm": {"linear": "Linear Regression", "multiple": "Multiple Linear Regression", "polynomial": "Polynomial Regression"}[algorithm],
            "target": target, "features": features, "degree": int(degree) if algorithm == "polynomial" else None,
            "coefficients": coefficients, "intercept": intercept,
            "actual": [float(v) for v in y_test.to_numpy()[:200]],
            "predicted": [float(v) for v in np.asarray(predictions)[:200]],
            "metrics": metrics, "cross_validation": cv, "complete_rows": int(len(X)),
        }
    record_regression_run(result["algorithm"], dataset_id, target, {**result["metrics"], "actual": result["actual"][:100], "predicted": result["predicted"][:100]})
    return result
