from __future__ import annotations
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.database.connection import optional_session
from backend.database.models import Device, SensorReading
from backend.services.database_service import record_prediction
from backend.utils.model_loader import predict_one


def _lookup_device(session: Session, device_key: str) -> Device | None:
    return session.scalar(select(Device).where((Device.id == device_key) | (Device.device_code == device_key)))


def register_device(device_id: str, device_name: str, location: str | None) -> dict:
    session = optional_session()
    if session is None:
        raise HTTPException(status_code=503, detail="PostgreSQL is required for device registration. Configure DATABASE_URL and confirm the database is available.")
    try:
        existing = _lookup_device(session, device_id)
        if existing:
            raise HTTPException(status_code=409, detail="A device with this device_id is already registered.")
        device = Device(id=str(uuid4()), device_code=device_id, device_name=device_name, location=location, status="offline")
        session.add(device)
        session.commit()
        session.refresh(device)
        return {"id": device.id, "device_id": device.device_code, "device_name": device.device_name, "location": device.location, "status": device.status, "created_at": device.created_at.isoformat(), "last_seen": None}
    except HTTPException:
        session.rollback()
        raise
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="Device id must be unique.") from exc
    finally:
        session.close()


def list_devices() -> list[dict]:
    session = optional_session()
    if session is None:
        raise HTTPException(status_code=503, detail="PostgreSQL is required for device monitoring. Configure DATABASE_URL.")
    try:
        devices = session.scalars(select(Device).order_by(Device.device_name)).all()
        threshold = datetime.now(timezone.utc) - timedelta(minutes=10)
        return [{"id": d.id, "device_id": d.device_code, "device_name": d.device_name, "location": d.location,
                 "status": "online" if d.last_seen and d.last_seen >= threshold else "offline",
                 "created_at": d.created_at.isoformat(), "last_seen": d.last_seen.isoformat() if d.last_seen else None} for d in devices]
    finally:
        session.close()


def get_device(device_key: str) -> dict:
    session = optional_session()
    if session is None:
        raise HTTPException(status_code=503, detail="PostgreSQL is required for device monitoring. Configure DATABASE_URL.")
    try:
        device = _lookup_device(session, device_key)
        if not device:
            raise HTTPException(status_code=404, detail="Device was not found.")
        latest = session.scalar(select(SensorReading).where(SensorReading.device_id == device.id).order_by(desc(SensorReading.timestamp)).limit(1))
        threshold = datetime.now(timezone.utc) - timedelta(minutes=10)
        live_status = "online" if device.last_seen and device.last_seen >= threshold else "offline"
        return {"id": device.id, "device_id": device.device_code, "device_name": device.device_name, "location": device.location,
                "status": live_status, "created_at": device.created_at.isoformat(), "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                "latest_reading": {"id": latest.id, "timestamp": latest.timestamp.isoformat(), "features": latest.water_quality_features, "source": latest.source} if latest else None}
    finally:
        session.close()


def get_readings(device_key: str, limit: int = 100) -> list[dict]:
    session = optional_session()
    if session is None:
        raise HTTPException(status_code=503, detail="PostgreSQL is required for reading history. Configure DATABASE_URL.")
    try:
        device = _lookup_device(session, device_key)
        if not device:
            raise HTTPException(status_code=404, detail="Device was not found.")
        readings = session.scalars(select(SensorReading).where(SensorReading.device_id == device.id).order_by(desc(SensorReading.timestamp)).limit(max(1, min(limit, 1000)))).all()
        return [{"id": r.id, "timestamp": r.timestamp.isoformat(), "features": r.water_quality_features, "source": r.source} for r in readings]
    finally:
        session.close()


def create_reading(device_key: str, timestamp: datetime | None, features: dict[str, float], source: str) -> dict:
    session = optional_session()
    if session is None:
        raise HTTPException(status_code=503, detail="PostgreSQL is required to persist IoT readings. Configure DATABASE_URL.")
    try:
        device = _lookup_device(session, device_key)
        if not device:
            raise HTTPException(status_code=404, detail="Device was not found. Register it before sending readings.")
        recorded_at = timestamp or datetime.now(timezone.utc)
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
        if len(features) > 300 or any(not isinstance(v, (int, float)) for v in features.values()):
            raise HTTPException(status_code=422, detail="Sensor features must be a numeric JSON object.")
        reading = SensorReading(id=str(uuid4()), device_id=device.id, timestamp=recorded_at, water_quality_features=features, source=source)
        device.last_seen = datetime.now(timezone.utc)
        device.status = "online"
        session.add(reading)
        session.commit()
        prediction = None
        prediction_error = None
        try:
            prediction = predict_one(features)
            record_prediction(source="iot", model_name=prediction["model_name"], prediction=prediction["prediction"], confidence=prediction.get("confidence"), features=features, device_id=device.id)
        except HTTPException as exc:
            if exc.status_code == 503:
                prediction_error = str(exc.detail)
            else:
                # The sensor reading is still valid historical telemetry even if its feature set doesn't match the model.
                prediction_error = str(exc.detail)
        return {"reading_id": reading.id, "device_id": device.device_code, "timestamp": recorded_at.isoformat(), "source": source,
                "features": features, "prediction": prediction, "prediction_status": "available" if prediction else "not_available", "prediction_message": prediction_error}
    except HTTPException:
        session.rollback()
        raise
    except SQLAlchemyError as exc:
        session.rollback()
        raise HTTPException(status_code=503, detail="Could not persist this reading in PostgreSQL.") from exc
    finally:
        session.close()
