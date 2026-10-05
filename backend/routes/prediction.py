from __future__ import annotations
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.database.connection import require_db
from backend.config import MAX_UPLOAD_BYTES
from backend.database.models import Prediction
from backend.schemas.prediction import PredictionRequest
from backend.services.dataset_service import load_dataset, save_uploaded_bytes
from backend.services.database_service import record_prediction, record_predictions_bulk, persist_dataset
from backend.services.ml_service import predict_frame, predict_one
from backend.utils.dataset_validator import json_safe
from sqlalchemy import desc, select

router = APIRouter()

@router.post("/predict")
def predict(request: PredictionRequest):
    result = predict_one(request.features)
    record_id = record_prediction(source=request.source[:24], model_name=result["model_name"], prediction=result["prediction"], confidence=result.get("confidence"), features=request.features)
    result["prediction_id"] = record_id
    result["persisted"] = bool(record_id)
    return result


@router.post("/predict-file")
async def predict_file(dataset_id: str | None = Form(default=None), file: UploadFile | None = File(default=None)):
    if bool(dataset_id) == bool(file):
        raise HTTPException(status_code=422, detail="Provide exactly one of dataset_id or file.")
    if file is not None:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        info = save_uploaded_bytes(raw, file.filename or "prediction.csv", is_demo=False)
        persist_dataset(info)
        dataset_id = info["id"]
        frame = pd.read_csv(info["_path"], low_memory=False)
        metadata = info
    else:
        frame, metadata = load_dataset(str(dataset_id))
    if len(frame) > 100_000:
        raise HTTPException(status_code=413, detail="File prediction is limited to 100,000 rows per request.")
    result = predict_frame(frame)
    predictions = result["predictions"]
    confidences = result.get("confidence")
    items = []
    persistence_rows = []
    for index, prediction in enumerate(predictions):
        row_features = {str(k): json_safe(v) for k, v in frame.iloc[index].to_dict().items()}
        confidence = confidences[index] if confidences is not None and index < len(confidences) else None
        items.append({"row": index + 1, "prediction": prediction, "confidence": confidence})
        # Keep write volume bounded while retaining the full computed response.
        if index < 5_000:
            persistence_rows.append({"prediction": prediction, "confidence": confidence, "features": row_features})
    persisted = record_predictions_bulk(source="dataset", model_name=result["model_name"], values=persistence_rows, dataset_id=dataset_id)
    return {"dataset_id": dataset_id, "file_name": metadata.get("file_name"), "row_count": len(items), "model_name": result["model_name"], "confidence_method": result.get("confidence_method"), "results": items, "persisted_predictions": persisted, "original_preserved": True}


@router.get("/predictions")
def predictions(limit: int = 100, _session=__import__("fastapi").Depends(require_db)):
    safe_limit = max(1, min(limit, 1000))
    rows = _session.scalars(select(Prediction).order_by(desc(Prediction.timestamp)).limit(safe_limit)).all()
    return {"items": [{"id": row.id, "source": row.source, "dataset_id": row.dataset_id, "device_id": row.device_id,
                       "timestamp": row.timestamp.isoformat(), "model_name": row.model_name, "prediction": row.prediction,
                       "confidence": row.confidence, "features": row.input_features} for row in rows], "count": len(rows)}
