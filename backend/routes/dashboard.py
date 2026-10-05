from __future__ import annotations
from sqlalchemy import desc, func, select
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter

from backend.database.connection import database_status, optional_session
from backend.database.models import ClusteringRun, DatasetRecord, Device, ModelRun, Prediction, RegressionRun, SensorReading, PCARun
from backend.services.ml_service import model_status

router = APIRouter()

@router.get("/dashboard")
def dashboard():
    db_state = database_status()
    model = model_status()
    session = optional_session()
    if session is None:
        return {
            "persistence_available": False,
            "database": db_state,
            "production_model": model,
            "counts": {"devices": None, "online_devices": None, "readings": None, "datasets": None, "predictions": None, "model_runs": None, "regression_runs": None, "clustering_runs": None, "pca_runs": None},
            "latest_readings": [], "latest_predictions": [], "recent_model_runs": [], "recent_regression_runs": [], "recent_clustering_runs": [], "recent_pca_runs": [],
            "message": "Dashboard history requires a reachable PostgreSQL database. Live counts are intentionally not fabricated.",
        }
    try:
        online_after = datetime.now(timezone.utc) - timedelta(minutes=10)
        counts = {
            "devices": session.scalar(select(func.count()).select_from(Device)) or 0,
            "online_devices": session.scalar(select(func.count()).select_from(Device).where(Device.last_seen >= online_after)) or 0,
            "readings": session.scalar(select(func.count()).select_from(SensorReading)) or 0,
            "datasets": session.scalar(select(func.count()).select_from(DatasetRecord)) or 0,
            "predictions": session.scalar(select(func.count()).select_from(Prediction)) or 0,
            "model_runs": session.scalar(select(func.count()).select_from(ModelRun)) or 0,
            "regression_runs": session.scalar(select(func.count()).select_from(RegressionRun)) or 0,
            "clustering_runs": session.scalar(select(func.count()).select_from(ClusteringRun)) or 0,
            "pca_runs": session.scalar(select(func.count()).select_from(PCARun)) or 0,
        }
        readings = session.scalars(select(SensorReading).order_by(desc(SensorReading.timestamp)).limit(8)).all()
        predictions = session.scalars(select(Prediction).order_by(desc(Prediction.timestamp)).limit(8)).all()
        models = session.scalars(select(ModelRun).order_by(desc(ModelRun.created_at)).limit(5)).all()
        regressions = session.scalars(select(RegressionRun).order_by(desc(RegressionRun.created_at)).limit(5)).all()
        clusters = session.scalars(select(ClusteringRun).order_by(desc(ClusteringRun.created_at)).limit(5)).all()
        pca_runs = session.scalars(select(PCARun).order_by(desc(PCARun.created_at)).limit(5)).all()
        return {
            "persistence_available": True, "database": db_state, "production_model": model, "counts": counts,
            "latest_readings": [{"id": r.id, "device_id": r.device_id, "timestamp": r.timestamp.isoformat(), "features": r.water_quality_features, "source": r.source} for r in readings],
            "latest_predictions": [{"id": p.id, "timestamp": p.timestamp.isoformat(), "prediction": p.prediction, "confidence": p.confidence, "model_name": p.model_name, "source": p.source} for p in predictions],
            "recent_model_runs": [{"id": r.id, "model_name": r.model_name, "metrics": r.metrics, "created_at": r.created_at.isoformat()} for r in models],
            "recent_regression_runs": [{"id": r.id, "algorithm": r.algorithm, "target": r.target, "metrics": r.metrics, "created_at": r.created_at.isoformat()} for r in regressions],
            "recent_clustering_runs": [{"id": r.id, "algorithm": r.algorithm, "cluster_count": r.cluster_count, "metrics": r.metrics, "created_at": r.created_at.isoformat()} for r in clusters],
            "recent_pca_runs": [{"id": r.id, "component_count": r.component_count, "metrics": r.metrics, "created_at": r.created_at.isoformat()} for r in pca_runs],
        }
    finally:
        session.close()
