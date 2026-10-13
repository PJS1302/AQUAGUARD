from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


_EMPTY_VALUES = {"", "na", "n/a", "nan", "null", "none", "nat"}
_IDENTIFIER_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|code|serial|identifier|record)([_\s-]|$)", re.I)
_DATE_NAME = re.compile(r"(date|time|timestamp|datetime)", re.I)
_TRUE_VALUES = {"true", "yes", "y", "1"}
_FALSE_VALUES = {"false", "no", "n", "0"}


def _clean_missing(series: pd.Series) -> pd.Series:
    result = series.copy()
    if pd.api.types.is_object_dtype(result.dtype) or isinstance(result.dtype, pd.StringDtype):
        text = result.astype("string").str.strip()
        result = result.mask(text.str.lower().isin(_EMPTY_VALUES), pd.NA)
    return result


def profile_column(name: str, series: pd.Series) -> dict[str, Any]:
    """Infer a column's usable role without changing the source series."""
    cleaned = _clean_missing(series)
    present = cleaned.dropna()
    missing_count = int(cleaned.isna().sum())
    profile: dict[str, Any] = {
        "name": str(name),
        "missing_values": missing_count,
        "invalid_values": 0,
        "infinite_values": 0,
        "unique_values": int(present.nunique(dropna=True)),
    }
    if present.empty:
        profile["role"] = "empty"
        return profile

    values = present.astype(str).str.strip()
    lower = values.str.lower()
    boolean_values = set(lower.unique())
    if pd.api.types.is_bool_dtype(series.dtype) or (
        not pd.api.types.is_numeric_dtype(series.dtype)
        and boolean_values.issubset(_TRUE_VALUES | _FALSE_VALUES)
    ):
        profile["role"] = "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        profile["role"] = "datetime"
    elif _DATE_NAME.search(str(name)) and not pd.api.types.is_numeric_dtype(series.dtype):
        parsed = pd.to_datetime(present, errors="coerce", utc=True)
        if float(parsed.notna().mean()) >= 0.7:
            profile["role"] = "datetime"
            profile["invalid_values"] = int(parsed.isna().sum())
        else:
            profile["role"] = "categorical"
    elif pd.api.types.is_numeric_dtype(series.dtype):
        profile["role"] = "numerical"
        numeric = pd.to_numeric(present, errors="coerce").to_numpy(dtype=float)
        profile["infinite_values"] = int(np.isinf(numeric).sum())
    else:
        numeric = pd.to_numeric(present, errors="coerce")
        numeric_ratio = float(numeric.notna().mean())
        if numeric_ratio >= 0.8:
            profile["role"] = "numerical"
            profile["invalid_values"] = int(numeric.isna().sum())
            finite = numeric.dropna().to_numpy(dtype=float)
            profile["infinite_values"] = int(np.isinf(finite).sum())
        else:
            date_shaped = present.astype(str).str.contains(r"[-/:T]", regex=True).mean() >= 0.7
            parsed = pd.to_datetime(present, errors="coerce", utc=True) if date_shaped else None
            if parsed is not None and float(parsed.notna().mean()) >= 0.8:
                profile["role"] = "datetime"
                profile["invalid_values"] = int(parsed.isna().sum())
            else:
                profile["role"] = "categorical"

    role = profile["role"]
    unique_ratio = profile["unique_values"] / max(1, len(present))
    if role != "empty" and (
        _IDENTIFIER_NAME.search(str(name))
        or (role == "categorical" and profile["unique_values"] >= 20 and unique_ratio >= 0.98)
    ):
        profile["role"] = "identifier"
    return profile


def profile_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [profile_column(str(name), frame[name]) for name in frame.columns]


def normalize_feature_frame(
    frame: pd.DataFrame,
    features: list[str],
    feature_roles: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize selected feature representations; learned fills/encodings happen in fitted estimators."""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Selected input columns are missing.", "missing_features": missing})
    if not features or len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select one or more distinct input features.")

    result = pd.DataFrame(index=frame.index)
    profiles: list[dict[str, Any]] = []
    for name in features:
        source = _clean_missing(frame[name])
        profile = profile_column(name, frame[name])
        role = (feature_roles or {}).get(name, profile["role"])
        profile["role"] = role
        if role == "empty":
            raise HTTPException(status_code=422, detail=f"Input feature '{name}' has no observed values; remove it or provide valid source data.")
        if role == "identifier":
            raise HTTPException(status_code=422, detail=f"Input feature '{name}' appears to be an identifier, not a measurement; exclude it unless it carries meaningful predictive information.")
        if role == "numerical":
            converted = pd.to_numeric(source, errors="coerce")
            numeric = converted.to_numpy(dtype=float, na_value=np.nan)
            numeric[~np.isfinite(numeric)] = np.nan
            result[name] = numeric
        elif role == "datetime":
            parsed = pd.to_datetime(source, errors="coerce", utc=True)
            numeric = parsed.astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000
            numeric[parsed.isna().to_numpy()] = np.nan
            result[name] = numeric
        elif role == "boolean":
            normalized = source.astype("string").str.strip().str.lower()
            result[name] = normalized.map(
                lambda value: True if value in _TRUE_VALUES else False if value in _FALSE_VALUES else pd.NA
            ).astype("object")
            profile["invalid_values"] = int(result[name].isna().sum()) - profile["missing_values"]
        else:
            result[name] = source.astype("string").astype("object")
        profiles.append(profile)

    diagnostics = {
        "columns": profiles,
        "imputation": "Numerical features use the training-partition median; categorical and Boolean features use the training-partition most-frequent value. Target values are never imputed.",
        "categorical_encoding": "One-hot encoding is fitted on training data; unseen prediction categories are ignored by the fitted encoder.",
        "date_time_encoding": "Date/time values are converted to UTC epoch seconds; unparseable values are treated as missing and are reported.",
        "source_rows": int(len(frame)),
        "application_dropped_rows": 0,
        "source_data_modified": False,
    }
    return result, diagnostics


class DatasetFeatureNormalizer(BaseEstimator, TransformerMixin):
    """Apply fixed type coercions while leaving learned transforms to later fitted steps."""

    def __init__(self, features: list[str], feature_roles: dict[str, str]):
        self.features = features
        self.feature_roles = feature_roles

    def fit(self, X: pd.DataFrame, y: Any = None):
        missing = [name for name in self.features if name not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing fitted feature(s): {missing}")
        self.feature_names_in_ = np.asarray(self.features, dtype=object)
        self.n_features_in_ = len(self.features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        normalized, _ = normalize_feature_frame(X, self.features, self.feature_roles)
        return normalized

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.features, dtype=object)


def make_preprocessing_pipeline(
    X: pd.DataFrame,
    estimator: Any,
    feature_roles: dict[str, str] | None = None,
) -> Pipeline:
    roles = feature_roles or {item["name"]: item["role"] for item in profile_frame(X)}
    type_normalizer = DatasetFeatureNormalizer([str(name) for name in X.columns], roles)
    empty_training_columns = [
        str(name)
        for name in X.columns
        if normalize_feature_frame(X, [str(name)], roles)[0][str(name)].isna().all()
    ]
    if empty_training_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "The training partition contains no observed values for selected features; no safe training-fitted imputation value exists.",
                "features": empty_training_columns,
            },
        )
    numeric = [str(name) for name in X.columns if roles.get(str(name)) in {"numerical", "datetime"}]
    categorical = [str(name) for name in X.columns if roles.get(str(name)) in {"categorical", "boolean"}]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise HTTPException(status_code=422, detail="No usable input features remain after type detection.")
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("type_normalizer", type_normalizer),
            ("preprocessing", preprocessor),
            ("estimator", estimator),
        ]
    )


def transform_unsupervised(frame: pd.DataFrame, features: list[str], *, allow_categorical: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    values, diagnostics = normalize_feature_frame(frame, features)
    if not allow_categorical:
        categorical = [name for name in features if not pd.api.types.is_numeric_dtype(values[name])]
        if categorical:
            raise HTTPException(status_code=422, detail={"message": "PCA requires numerical features; select numeric or date/time columns.", "categorical_features": categorical})
    for name in values.columns:
        if values[name].isna().all():
            raise HTTPException(status_code=422, detail=f"Input feature '{name}' has no valid values.")
    if values.isna().any().any():
        if allow_categorical:
            numeric = values.select_dtypes(include=[np.number]).columns.tolist()
            categorical = [name for name in values.columns if name not in numeric]
            transformers = []
            if numeric:
                transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
            if categorical:
                transformers.append(
                    (
                        "categorical",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        categorical,
                    )
                )
            transform = ColumnTransformer(transformers, sparse_threshold=0, verbose_feature_names_out=False)
        else:
            transform = SimpleImputer(strategy="median")
        values = pd.DataFrame(transform.fit_transform(values), index=values.index)
        transformed_names = (
            transform.get_feature_names_out().tolist()
            if hasattr(transform, "get_feature_names_out")
            else [str(name) for name in features]
        )
        diagnostics["fitted_imputation"] = "Fit on the rows used by this unsupervised analysis."
    else:
        diagnostics["fitted_imputation"] = "No feature values required imputation."
        if allow_categorical:
            numeric = values.select_dtypes(include=[np.number]).columns.tolist()
            categorical = [name for name in values.columns if name not in numeric]
            if categorical:
                transform = ColumnTransformer(
                    [
                        ("numeric", "passthrough", numeric),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ],
                    sparse_threshold=0,
                    verbose_feature_names_out=False,
                )
                values = pd.DataFrame(transform.fit_transform(values), index=values.index)
                transformed_names = transform.get_feature_names_out().tolist()
            else:
                transformed_names = [str(name) for name in features]
        else:
            transformed_names = [str(name) for name in features]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise HTTPException(status_code=422, detail="Preprocessing left non-finite values; the selected features cannot be safely transformed.")
    diagnostics["applied_transforms"] = ["training-data imputation"] + (["one-hot categorical encoding"] if allow_categorical else [])
    diagnostics["transformed_feature_count"] = int(values.shape[1])
    diagnostics["transformed_feature_names"] = transformed_names
    return values, diagnostics


def require_complete_processed_rows(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    task: str,
    minimum: int = 1,
) -> pd.DataFrame:
    """Retained for callers that explicitly require complete cases; never used as a universal row rule."""
    selected = frame.loc[:, columns].copy(deep=True)
    complete = ~selected.isna().any(axis=1)
    count = int(complete.sum())
    if count < minimum:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"{task} needs at least {minimum} complete row(s) for this operation; {count} are available.",
                "processed_rows": int(len(selected)),
                "complete_processed_rows": count,
                "incomplete_processed_rows": int((~complete).sum()),
            },
        )
    return selected.loc[complete].copy()


def require_finite_numeric(frame: pd.DataFrame, columns: list[str], *, task: str) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise HTTPException(status_code=422, detail=f"{task} requires numerical input '{column}'.")
        try:
            finite = np.isfinite(frame[column].to_numpy(dtype=float)).all()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Input '{column}' is not numerical.") from exc
        if not finite:
            raise HTTPException(status_code=422, detail=f"Input '{column}' contains non-finite values.")
