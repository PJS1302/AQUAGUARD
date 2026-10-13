from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc, select

from backend.database.connection import optional_session
from backend.database.models import DatasetRecord
from backend.config import MAX_UPLOAD_BYTES
from backend.services.dataset_service import (
    get_dataset_path, get_processed_path, inspect_dataframe, list_local_datasets,
    load_dataset, save_processed_copy, save_uploaded_bytes,
)
from backend.services.user_preprocessing_service import apply_user_preprocessing
from backend.services.database_service import persist_dataset
from backend.services.preprocessing_service import data_quality_report
from backend.services.ml_service import model_status

router = APIRouter()


def _compatibility(columns: list[str], frame: pd.DataFrame | None = None) -> dict[str, Any]:
    status = model_status()
    if not status.get("available"):
        return {
            "status": "no_production_model",
            "schema_compatible": False,
            "prediction_ready": False,
            "reason": status.get("reason") or "No production model artifact is installed.",
            "missing_features": [],
            "missing_values_count": 0,
            "preprocessing_connected": False,
        }
    expected = status.get("features", [])
    missing = [name for name in expected if name not in columns]
    schema_compatible = len(missing) == 0

    has_missing_values = False
    missing_values_count = 0
    incompatible_types = []
    if frame is not None and schema_compatible:
        for f in expected:
            if f in frame.columns:
                n_miss = int(frame[f].isna().sum())
                missing_values_count += n_miss
                if n_miss > 0:
                    has_missing_values = True
                # Check numeric conversion
                numeric = pd.to_numeric(frame[f].dropna(), errors="coerce")
                if numeric.isna().any():
                    incompatible_types.append(f)

    preprocessing_connected = bool(status.get("preprocessing_connected"))
    
    # Prediction readiness requires:
    # 1. Schema compatible (all expected features present)
    # 2. No incompatible types
    # 3. Either no missing values OR a preprocessing pipeline is connected to handle missing values
    prediction_ready = (
        schema_compatible
        and len(incompatible_types) == 0
        and (not has_missing_values or preprocessing_connected)
    )

    if not schema_compatible:
        status_text = "missing_features: " + ", ".join(missing)
    elif incompatible_types:
        status_text = "incompatible_types: " + ", ".join(incompatible_types)
    elif has_missing_values and not preprocessing_connected:
        status_text = "missing_values_without_preprocessor"
    else:
        status_text = "compatible_feature_schema"

    return {
        "status": status_text,
        "schema_compatible": schema_compatible,
        "prediction_ready": prediction_ready,
        "missing_features": missing,
        "missing_values_count": missing_values_count,
        "incompatible_types": incompatible_types,
        "preprocessing_connected": preprocessing_connected,
        "model_name": status.get("model_name"),
    }


def _public_info(info: dict, frame: pd.DataFrame | None = None) -> dict:
    result = {k: v for k, v in info.items() if not k.startswith("_")}
    compat = _compatibility(result.get("columns", []), frame)
    result["compatibility_status"] = compat["status"]
    result["prediction_readiness"] = compat
    return result


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), is_demo: bool = Form(False)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    info = save_uploaded_bytes(raw, file.filename or "dataset.csv", is_demo=is_demo)
    frame = __import__("pandas").read_csv(info["_path"], low_memory=False)
    compat = _compatibility(info["columns"], frame)
    info["compatibility_status"] = compat["status"]
    info["prediction_readiness"] = compat
    sidecar = Path(info["_path"]).parent / "metadata.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["compatibility_status"] = info["compatibility_status"]
    metadata["prediction_readiness"] = compat
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    persisted = persist_dataset(info)
    public = _public_info(info, frame)
    public["postgres_persisted"] = persisted
    public["data_quality"] = data_quality_report(frame)
    return public


@router.get("/datasets")
def list_datasets():
    session = optional_session()
    if session is None:
        local = list_local_datasets()
        return {"items": local, "postgres_available": False, "message": "Dataset files remain on the backend filesystem; configure PostgreSQL to persist dataset metadata."}
    try:
        records = session.scalars(select(DatasetRecord).order_by(desc(DatasetRecord.upload_time))).all()
        return {"items": [{"id": r.id, "file_name": r.file_name, "uploaded_at": r.upload_time.isoformat(), "row_count": r.row_count, "column_count": r.column_count,
                            "compatibility_status": r.compatibility_status, "missing_value_summary": r.missing_value_summary, "is_demo": r.is_demo, "columns": r.columns} for r in records], "postgres_available": True}
    finally:
        session.close()


@router.get("/datasets/{dataset_id}/original")
def download_original(dataset_id: str):
    path = get_dataset_path(dataset_id)
    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.get("/datasets/{dataset_id}")
def inspect_dataset(dataset_id: str):
    frame, metadata = load_dataset(dataset_id)
    compat = _compatibility([str(c) for c in frame.columns], frame)
    return {
        **{k: v for k, v in metadata.items() if k != "preview"},
        **inspect_dataframe(frame),
        "data_quality": data_quality_report(frame),
        "compatibility_status": compat["status"],
        "prediction_readiness": compat,
        "original_preserved": True,
    }


@router.post("/datasets/{dataset_id}/preprocess")
def preprocess_dataset(dataset_id: str):
    original, _ = load_dataset(dataset_id)
    # Only the locally installed, user-authored pipeline is allowed to transform this copy.
    processed, report = apply_user_preprocessing(original)
    process_id = save_processed_copy(processed)
    result = inspect_dataframe(processed)
    return {
        "process_id": process_id,
        "download_url": f"/datasets/processed/{process_id}",
        "report": report,
        "processed": result,
        "original_preserved": True,
    }


@router.get("/datasets/processed/{process_id}")
def download_processed(process_id: str):
    path = get_processed_path(process_id)
    return FileResponse(path, media_type="text/csv", filename=f"aquaguard_processed_{process_id[:8]}.csv")
