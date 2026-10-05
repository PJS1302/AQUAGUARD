from __future__ import annotations
import json
from pathlib import Path

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


def _compatibility(columns: list[str]) -> str:
    status = model_status()
    if not status.get("available"):
        return "no_production_model"
    missing = [name for name in status.get("features", []) if name not in columns]
    return "missing_features: " + ", ".join(missing) if missing else "compatible_feature_schema"


def _public_info(info: dict) -> dict:
    result = {k: v for k, v in info.items() if not k.startswith("_")}
    result["compatibility_status"] = _compatibility(result.get("columns", []))
    return result


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), is_demo: bool = Form(False)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    info = save_uploaded_bytes(raw, file.filename or "dataset.csv", is_demo=is_demo)
    info["compatibility_status"] = _compatibility(info["columns"])
    sidecar = Path(info["_path"]).parent / "metadata.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["compatibility_status"] = info["compatibility_status"]
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    persisted = persist_dataset(info)
    public = _public_info(info)
    public["postgres_persisted"] = persisted
    public["data_quality"] = data_quality_report(__import__("pandas").read_csv(info["_path"], low_memory=False))
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
    return {**{k: v for k, v in metadata.items() if k != "preview"}, **inspect_dataframe(frame), "data_quality": data_quality_report(frame), "compatibility_status": _compatibility([str(c) for c in frame.columns]), "original_preserved": True}


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
