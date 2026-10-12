from __future__ import annotations

from fastapi import HTTPException

from backend.services.dataset_service import load_dataset
from backend.services.user_preprocessing_service import preprocessing_status


def load_preprocessed_dataset(dataset_id: str, expected_sha256: str | None = None):
    """Load original data; fitted transforms are created by each task's estimator."""
    original, metadata = load_dataset(dataset_id)
    artifact_status = preprocessing_status()
    if expected_sha256 and expected_sha256 != artifact_status.get("artifact_sha256"):
        raise HTTPException(
            status_code=409,
            detail="The configured preprocessing artifact changed after the experiment was configured. Review preprocessing settings and rerun.",
        )
    preprocessing_report = {
        "pipeline_name": "AquaGuard dataset-aware fitted preprocessing",
        "pipeline_steps": [
            "detect selected feature data types",
            "coerce invalid numeric/date values to missing with diagnostics",
            "fit numerical imputation and categorical encoding on training data only",
            "reuse fitted transformations for holdout, cross-validation, and saved experiment predictions",
        ],
        "configured_artifact_sha256": artifact_status.get("artifact_sha256"),
        "configured_artifact_available": bool(artifact_status.get("available")),
        "configured_frame_pipeline_used_for_training": False,
        "source_rows": int(len(original)),
        "source_data_modified": False,
        "note": "The original upload is passed through unchanged. Task-specific transformations are fitted inside each model on training data only.",
    }
    return original, metadata, preprocessing_report
