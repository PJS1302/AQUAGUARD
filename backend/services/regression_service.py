from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from backend.services.database_service import record_regression_run
from backend.utils.experiment_data import make_preprocessing_pipeline, normalize_feature_frame


def prepare_regression_data(
    frame: pd.DataFrame, target: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if target not in frame.columns:
        raise HTTPException(status_code=422, detail=f"Target column '{target}' was not found in the dataset.")
    if not features or len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select one or more distinct input features.")
    if target in features:
        raise HTTPException(status_code=422, detail="The target cannot also be an input feature.")
    normalized, diagnostics = normalize_feature_frame(frame, features)

    raw_target = frame[target]
    if raw_target.dtype == object or isinstance(raw_target.dtype, pd.StringDtype):
        text = raw_target.astype("string").str.strip()
        raw_target = raw_target.mask(text.str.lower().isin({"", "na", "n/a", "nan", "null", "none", "nat"}), pd.NA)
    y = pd.to_numeric(raw_target, errors="coerce")
    values = y.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values)
    excluded = int((~valid).sum())
    X = frame.loc[valid, features].copy(deep=True)
    normalized = normalized.loc[valid].copy()
    y = pd.Series(values[valid], index=X.index, name=target, dtype=float)
    if y.nunique() < 2:
        raise HTTPException(
            status_code=422,
            detail="Regression needs a continuous numeric target with at least two distinct observed values; this target appears constant or has too few valid observations.",
        )
    diagnostics["target"] = {
        "name": target,
        "invalid_or_missing_rows_excluded": excluded,
        "imputed": False,
        "reason": "Supervised regression cannot train on rows without a valid numeric target.",
    }
    diagnostics["rows_used"] = int(len(X))
    diagnostics["feature_missing_values"] = {
        str(name): int(normalized[name].isna().sum())
        for name in normalized.columns
        if normalized[name].isna().any()
    }
    diagnostics["feature_roles"] = {item["name"]: item["role"] for item in diagnostics["columns"]}
    return X, y, diagnostics


def _metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def _cross_validation(estimator, X: pd.DataFrame, y: pd.Series, folds: int) -> dict:
    usable_folds = min(int(folds), len(X) // 2)
    if usable_folds < 2:
        return {
            "folds": 0,
            "fold_scores": [],
            "mean_score": None,
            "std_dev": None,
            "metric": "R²",
            "message": "Cross-validation needs at least four valid target rows so every R² fold has at least two test samples.",
        }
    splitter = KFold(n_splits=usable_folds, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(estimator, X, y, cv=splitter, scoring="r2", error_score="raise")
    except Exception as exc:
        return {
            "folds": usable_folds,
            "fold_scores": [],
            "mean_score": None,
            "std_dev": None,
            "metric": "R²",
            "message": f"Cross-validation could not be calculated: {exc}",
        }
    return {
        "folds": usable_folds,
        "fold_scores": [float(score) for score in scores],
        "mean_score": float(np.mean(scores)),
        "std_dev": float(np.std(scores, ddof=0)),
        "metric": "R²",
    }


def _lwlr_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, bandwidth: float) -> np.ndarray:
    train = np.asarray(X_train, dtype=float)
    queries = np.asarray(X_query, dtype=float)
    design = np.column_stack([np.ones(len(train)), train])
    out = []
    for query in queries:
        query_design = np.r_[1.0, query]
        distance_sq = np.sum((train - query) ** 2, axis=1)
        weights = np.exp(-distance_sq / (2.0 * float(bandwidth) ** 2))
        sqrt_w = np.sqrt(np.maximum(weights, 1e-12))
        coefficients = np.linalg.lstsq(design * sqrt_w[:, None], y_train * sqrt_w, rcond=None)[0]
        out.append(float(query_design @ coefficients))
    return np.asarray(out, dtype=float)


def _lwlr_cv(
    X: pd.DataFrame,
    y: pd.Series,
    requested_folds: int,
    bandwidth: float,
    feature_roles: dict[str, str],
) -> dict:
    folds = min(int(requested_folds), len(X) // 2)
    if folds < 2:
        return {
            "folds": 0,
            "fold_scores": [],
            "mean_score": None,
            "std_dev": None,
            "metric": "R²",
            "message": "Cross-validation needs at least four valid target rows so every R² fold has at least two test samples.",
        }
    scores = []
    for train_idx, test_idx in KFold(n_splits=folds, shuffle=True, random_state=42).split(X):
        train = X.iloc[train_idx]
        test = X.iloc[test_idx]
        preprocessor = make_preprocessing_pipeline(train, "passthrough", feature_roles)
        transformed_train = preprocessor.fit_transform(train)
        transformed_test = preprocessor.transform(test)
        prediction = _lwlr_predict(
            transformed_train,
            y.iloc[train_idx].to_numpy(float),
            transformed_test,
            bandwidth,
        )
        scores.append(float(r2_score(y.iloc[test_idx], prediction)))
    return {
        "folds": folds,
        "fold_scores": scores,
        "mean_score": float(np.mean(scores)),
        "std_dev": float(np.std(scores, ddof=0)),
        "metric": "R²",
    }


def _model_pipeline(
    X_train: pd.DataFrame, algorithm: str, degree: int, feature_roles: dict[str, str]
) -> Pipeline:
    if algorithm == "polynomial":
        base = make_preprocessing_pipeline(X_train, "passthrough", feature_roles)
        return Pipeline(
            base.steps[:-1]
            + [
                ("polynomialfeatures", PolynomialFeatures(degree=degree, include_bias=False)),
                ("estimator", LinearRegression()),
            ]
        )
    return make_preprocessing_pipeline(X_train, LinearRegression(), feature_roles)


def run_regression(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    algorithm: str,
    target: str,
    features: list[str],
    degree: int = 2,
    folds: int = 5,
    bandwidth: float = 0.8,
    query: dict | None = None,
) -> dict:
    X, y, preprocessing = prepare_regression_data(frame, target, features)
    feature_roles = preprocessing["feature_roles"]
    if algorithm not in {"linear", "multiple", "polynomial", "lwlr"}:
        raise HTTPException(status_code=400, detail="Unknown regression algorithm.")
    if algorithm == "polynomial" and not (2 <= int(degree) <= 5):
        raise HTTPException(status_code=422, detail="Polynomial degree must be between 2 and 5.")
    if algorithm == "lwlr" and len(X) > 20_000:
        raise HTTPException(status_code=413, detail="LWLR is quadratic in the number of samples and is limited to 20,000 valid target rows per run.")
    if algorithm == "lwlr" and not (0 < float(bandwidth) <= 20):
        raise HTTPException(status_code=422, detail="Bandwidth must be greater than zero and no more than 20.")
    if len(X) < 2:
        raise HTTPException(
            status_code=422,
            detail="A holdout evaluation needs at least two valid target rows: one training row and one independent test row.",
        )

    n_test = min(len(X) - 1, max(1, int(math.ceil(len(X) * 0.2))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
    if algorithm == "lwlr":
        preprocessor = make_preprocessing_pipeline(X_train, "passthrough", feature_roles)
        try:
            X_train_fit = preprocessor.fit_transform(X_train)
            X_test_fit = preprocessor.transform(X_test)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Regression preprocessing could not be fitted: {exc}") from exc
        predictions = _lwlr_predict(X_train_fit, y_train.to_numpy(float), X_test_fit, bandwidth)
        metrics = _metrics(y_test, predictions)
        cv = _lwlr_cv(X, y, folds, bandwidth, feature_roles)
        if query:
            missing_query = [feature for feature in features if feature not in query]
            if missing_query:
                raise HTTPException(status_code=422, detail={"message": "LWLR query point is missing selected feature(s).", "missing_features": missing_query})
            query_frame = pd.DataFrame([query], columns=features)
            query_fit = preprocessor.transform(query_frame)
            query_prediction = float(_lwlr_predict(X_train_fit, y_train.to_numpy(float), query_fit, bandwidth)[0])
            query_point = {key: query[key] for key in features}
        else:
            query_point = X_test.iloc[0].to_dict()
            query_prediction = float(predictions[0])
        result = {
            "algorithm": "Locally Weighted Linear Regression (LWLR)",
            "target": target,
            "features": features,
            "bandwidth": float(bandwidth),
            "query_point": query_point,
            "query_prediction": query_prediction,
            "actual": [float(value) for value in y_test.to_numpy()[:200]],
            "predicted": [float(value) for value in predictions[:200]],
            "metrics": metrics,
            "cross_validation": cv,
            "complete_rows": int(len(X)),
            "target_rows_excluded": preprocessing["target"]["invalid_or_missing_rows_excluded"],
            "feature_preprocessing": preprocessing,
        }
    else:
        if algorithm == "polynomial":
            terms = math.comb(len(features) + int(degree), int(degree)) - 1
            if terms > 5_000:
                raise HTTPException(status_code=422, detail="Polynomial expansion would exceed 5,000 terms. Reduce feature count or degree.")
        estimator = _model_pipeline(X_train, algorithm, int(degree), feature_roles)
        try:
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Regression could not be fitted: {exc}") from exc
        metrics = _metrics(y_test, predictions)
        cv = _cross_validation(_model_pipeline(X, algorithm, int(degree), feature_roles), X, y, folds)
        preprocessor = estimator.named_steps["preprocessing"]
        feature_names = preprocessor.get_feature_names_out().tolist()
        if algorithm == "polynomial":
            names = estimator.named_steps["polynomialfeatures"].get_feature_names_out(feature_names).tolist()
        else:
            names = feature_names
        fitted = estimator.named_steps["estimator"]
        coefficients = {str(name): float(value) for name, value in zip(names, fitted.coef_)}
        result = {
            "algorithm": {"linear": "Linear Regression", "multiple": "Multiple Linear Regression", "polynomial": "Polynomial Regression"}[algorithm],
            "target": target,
            "features": features,
            "degree": int(degree) if algorithm == "polynomial" else None,
            "coefficients": coefficients,
            "intercept": float(fitted.intercept_),
            "actual": [float(value) for value in y_test.to_numpy()[:200]],
            "predicted": [float(value) for value in np.asarray(predictions)[:200]],
            "metrics": metrics,
            "cross_validation": cv,
            "complete_rows": int(len(X)),
            "target_rows_excluded": preprocessing["target"]["invalid_or_missing_rows_excluded"],
            "feature_preprocessing": preprocessing,
        }
    record_regression_run(
        result["algorithm"],
        dataset_id,
        target,
        {**result["metrics"], "actual": result["actual"][:100], "predicted": result["predicted"][:100]},
    )
    return result
