"""Adapter for the user's trusted, locally installed preprocessing implementation.

AquaGuard intentionally has no default cleaning or transformation recipe. The
artifact is loaded only from backend/models/ (never from a CSV upload).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import joblib
import pandas as pd
from fastapi import HTTPException

from backend.config import MODELS_DIR

ARTIFACT_PATH = MODELS_DIR / "user_preprocessing_pipeline.joblib"
ARTIFACT_FORMAT = "aquaguard.frame-preprocessor.v1"


def _unpack_artifact() -> tuple[Any | None, dict[str, Any], dict[str, Any]]:
    """Load and validate the trusted local adapter artifact, without fallbacks."""
    if not ARTIFACT_PATH.is_file():
        return None, {}, {
            "available": False,
            "reason": (
                "The user's Python preprocessing pipeline is not installed. "
                "AquaGuard will not drop rows, impute, scale, encode, or otherwise "
                "transform data until that exact pipeline is supplied."
            ),
            "artifact_file": str(ARTIFACT_PATH.relative_to(MODELS_DIR.parent)),
            "required_method": "preprocess_frame(DataFrame) -> DataFrame",
        }

    try:
        artifact = joblib.load(ARTIFACT_PATH)
    except Exception as exc:
        return None, {}, {
            "available": False,
            "reason": f"The installed preprocessing artifact could not be loaded: {exc}",
            "artifact_file": str(ARTIFACT_PATH.relative_to(MODELS_DIR.parent)),
            "required_method": "preprocess_frame(DataFrame) -> DataFrame",
        }

    metadata: dict[str, Any] = {}
    pipeline = artifact
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline")
        metadata = artifact.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None, {}, {
                "available": False,
                "reason": "The preprocessing artifact's 'metadata' entry must be an object.",
                "artifact_file": str(ARTIFACT_PATH.relative_to(MODELS_DIR.parent)),
                "required_method": "preprocess_frame(DataFrame) -> DataFrame",
            }

    method = getattr(pipeline, "preprocess_frame", None)
    if not callable(method):
        return None, {}, {
            "available": False,
            "reason": (
                "The installed artifact must expose preprocess_frame(frame), which "
                "returns the user's processed DataFrame and preserves source row indexes."
            ),
            "artifact_file": str(ARTIFACT_PATH.relative_to(MODELS_DIR.parent)),
            "required_method": "preprocess_frame(DataFrame) -> DataFrame",
        }

    steps = getattr(pipeline, "preprocessing_steps", metadata.get("steps", []))
    if not isinstance(steps, list) or not all(isinstance(item, str) for item in steps):
        steps = []
    name = getattr(pipeline, "pipeline_name", None) or metadata.get("name") or type(pipeline).__name__
    try:
        artifact_sha256 = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        return None, {}, {
            "available": False,
            "reason": f"The preprocessing artifact could not be fingerprinted: {exc}",
            "artifact_file": str(ARTIFACT_PATH.relative_to(MODELS_DIR.parent)),
            "required_method": "preprocess_frame(DataFrame) -> DataFrame",
        }
    status = {
        "artifact_exists": True,
        "available": True,
        "reason": None,
        "pipeline_name": str(name),
        "steps": steps,
        "artifact_file": str(ARTIFACT_PATH.relative_to(MODELS_DIR.parent)),
        "artifact_sha256": artifact_sha256,
        "format": metadata.get("format", ARTIFACT_FORMAT),
        "required_method": "preprocess_frame(DataFrame) -> DataFrame",
        "supports_production_inference": callable(getattr(pipeline, "transform_features", None)),
        "input_features": metadata.get("input_features", getattr(pipeline, "input_features", None)),
    }
    return pipeline, metadata, status


def preprocessing_status(dataset_id: str | None = None) -> dict[str, Any]:
    """Report preprocessing readiness across artifact existence, production model usage, and dataset transformation."""
    pipeline, _, status = _unpack_artifact()
    from backend.utils.model_loader import get_model_bundle
    bundle = get_model_bundle()
    status["production_model_uses_preprocessing"] = bool(
        bundle.get("available") and (bundle.get("integrated") or bundle.get("preprocessing") is not None)
    )
    status["production_model_name"] = bundle.get("model_name") if bundle.get("available") else None
    status["transforms_selected_dataset"] = None
    status["dataset_changed"] = None
    status["dataset_transform_error"] = None

    if dataset_id and status.get("available") and pipeline is not None:
        try:
            from backend.services.dataset_service import load_dataset
            frame, _ = load_dataset(dataset_id)
            input_copy = frame.copy(deep=True)
            processed = pipeline.preprocess_frame(input_copy)
            status["transforms_selected_dataset"] = isinstance(processed, pd.DataFrame)
            status["dataset_changed"] = bool(not processed.equals(frame) if isinstance(processed, pd.DataFrame) else False)
        except Exception as exc:
            status["transforms_selected_dataset"] = False
            status["dataset_transform_error"] = str(exc)

    return status


def get_user_pipeline() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    pipeline, metadata, status = _unpack_artifact()
    if not status.get("available") or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": status.get("reason", "The user's preprocessing pipeline is unavailable."),
                "preprocessing_status": status,
            },
        )
    return pipeline, metadata, status


def apply_user_preprocessing(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run only the locally installed user pipeline against an isolated copy.

    The artifact must preserve original DataFrame index labels on all retained
    rows. That lets AquaGuard verify that rows were excluded only by the user's
    pipeline and keeps target/feature alignment auditable.
    """
    pipeline, metadata, status = get_user_pipeline()
    if not frame.index.is_unique:
        raise HTTPException(status_code=422, detail="The source dataset has non-unique row indexes and cannot be aligned safely.")

    input_copy = frame.copy(deep=True)
    try:
        processed = pipeline.preprocess_frame(input_copy)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "The installed user preprocessing pipeline failed. AquaGuard did not "
                f"apply a fallback or additional preprocessing. Details: {exc}"
            ),
        ) from exc

    if not isinstance(processed, pd.DataFrame):
        raise HTTPException(
            status_code=422,
            detail="The user pipeline must return a pandas DataFrame so source rows, labels, and transformed columns remain traceable.",
        )
    if not processed.index.is_unique:
        raise HTTPException(status_code=422, detail="The user pipeline returned duplicate row indexes; rows cannot be aligned safely.")
    if not processed.index.isin(frame.index).all():
        raise HTTPException(
            status_code=422,
            detail="The user pipeline must preserve the source DataFrame index for each retained row; AquaGuard will not guess row alignment.",
        )
    if len(processed.columns) == 0:
        raise HTTPException(status_code=422, detail="The user pipeline returned no columns.")
    if processed.columns.has_duplicates:
        raise HTTPException(status_code=422, detail="The user pipeline returned duplicate column names.")
    if not all(isinstance(column, str) for column in processed.columns):
        raise HTTPException(status_code=422, detail="The user pipeline must return string column names for dataset selection and export.")

    processed = processed.copy(deep=True)
    complete_mask = ~processed.isna().any(axis=1) if len(processed.columns) else pd.Series(False, index=processed.index)
    missing_before = int(frame.isna().sum().sum())
    missing_after = int(processed.isna().sum().sum())
    data_changed = (missing_before != missing_after) or (len(processed) != len(frame)) or (not processed.equals(frame))

    if data_changed:
        note = "AquaGuard applied the configured pipeline to a separate copy; configured transformations were performed and missing values imputed."
    else:
        note = "The configured user pipeline processed the dataset and left values unchanged; no modifications or row removals were necessary."

    report = {
        "pipeline_name": status["pipeline_name"],
        "pipeline_steps": status.get("steps", []),
        "artifact_file": status["artifact_file"],
        "artifact_sha256": status["artifact_sha256"],
        "original_rows": int(len(frame)),
        "processed_rows": int(len(processed)),
        "rows_removed_by_pipeline": int(len(frame) - len(processed)),
        "complete_processed_rows": int(complete_mask.sum()),
        "incomplete_processed_rows": int((~complete_mask).sum()),
        "missing_cells_before": missing_before,
        "missing_cells_after": missing_after,
        "original_preserved": True,
        "application_added_preprocessing": False,
        "data_changed": bool(data_changed),
        "note": note,
    }
    return processed, report



class UserPipelinePredictor:
    """Production wrapper requiring the same artifact's explicit feature transform."""
    def __init__(self, pipeline: Any, estimator: Any, output_features: list[str]):
        self.pipeline = pipeline
        self.estimator = estimator
        self.output_features = list(output_features)
        self.classes_ = getattr(estimator, "classes_", None)
        self.n_features_in_ = len(self.output_features)
        self.feature_names_in_ = pd.Index(self.output_features).to_numpy(dtype=object)

    def _transform(self, X: Any):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.output_features)
        transform = getattr(self.pipeline, "transform_features", None)
        if not callable(transform):
            raise ValueError("The installed user pipeline does not implement transform_features for production inference.")
        transformed = transform(X.copy(deep=True))
        if not isinstance(transformed, pd.DataFrame):
            raise ValueError("transform_features must return a pandas DataFrame.")
        if not transformed.index.equals(X.index):
            raise ValueError("Production transform_features must preserve all input rows and their indexes.")
        missing = [name for name in self.output_features if name not in transformed.columns]
        if missing:
            raise ValueError(f"The preprocessing output is missing trained feature(s): {missing}")
        return transformed.loc[:, self.output_features]

    def predict(self, X: Any):
        return self.estimator.predict(self._transform(X))

    def predict_proba(self, X: Any):
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("The trained estimator does not expose predict_proba.")
        return self.estimator.predict_proba(self._transform(X))
