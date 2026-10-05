from __future__ import annotations

from backend.services.dataset_service import load_dataset
from fastapi import HTTPException

from backend.services.user_preprocessing_service import apply_user_preprocessing


def load_preprocessed_dataset(dataset_id: str, expected_sha256: str | None = None):
    """Load an immutable original and apply only the installed user pipeline copy."""
    original, metadata = load_dataset(dataset_id)
    processed, preprocessing_report = apply_user_preprocessing(original)
    if expected_sha256 and expected_sha256 != preprocessing_report.get("artifact_sha256"):
        raise HTTPException(
            status_code=409,
            detail="The user preprocessing pipeline changed after the processing copy was created. Recreate the copy and rerun the experiment.",
        )
    return processed, metadata, preprocessing_report
