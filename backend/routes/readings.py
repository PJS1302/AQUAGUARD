from __future__ import annotations
from datetime import datetime
import math

from fastapi import APIRouter, HTTPException, Request

from backend.services.device_service import create_reading, get_readings

router = APIRouter(prefix="/devices")

@router.get("/{device_id}/readings")
def readings(device_id: str, limit: int = 100):
    return {"items": get_readings(device_id, limit)}

@router.post("/{device_id}/readings")
async def post_reading(device_id: str, request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Reading body must be a JSON object.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Reading body must be a JSON object.")
    body_device = body.get("device_id")
    if body_device and str(body_device) != device_id:
        raise HTTPException(status_code=422, detail="Path device_id and body device_id do not match.")
    timestamp = body.get("timestamp")
    if timestamp is not None:
        if not isinstance(timestamp, str):
            raise HTTPException(status_code=422, detail="timestamp must be an ISO-8601 string.")
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="timestamp must be a valid ISO-8601 timestamp.") from exc
    source = body.get("source", "sensor")
    if source not in {"sensor", "simulation"}:
        raise HTTPException(status_code=422, detail="source must be 'sensor' or 'simulation'.")
    features = body.get("features")
    if features is None:
        reserved = {"device_id", "timestamp", "source"}
        features = {key: value for key, value in body.items() if key not in reserved}
    if not isinstance(features, dict) or not features:
        raise HTTPException(status_code=422, detail="Provide numeric features either inside 'features' or as flat JSON fields.")
    clean = {}
    for key, value in features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise HTTPException(status_code=422, detail=f"Sensor feature '{key}' must be a finite number.")
        clean[str(key)] = float(value)
    return create_reading(device_id, timestamp, clean, source)
