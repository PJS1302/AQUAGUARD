from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError

from backend.database.connection import optional_session
from backend.database.models import DatasetRecord, Prediction, ModelRun, RegressionRun, ClusteringRun, PCARun, Device, SensorReading


def persist_dataset(info: dict) -> bool:
    session = optional_session()
    if session is None:
        return False
    try:
        existing = session.get(DatasetRecord, info["id"])
        values = dict(
            file_name=info["file_name"], row_count=info["row_count"], column_count=info["column_count"],
            compatibility_status=info.get("compatibility_status", "inspection_only"),
            missing_value_summary=info.get("missing_value_summary", {}), is_demo=bool(info.get("is_demo")),
            columns=info.get("columns", []),
        )
        if existing is None:
            session.add(DatasetRecord(id=info["id"], **values))
        else:
            for key, value in values.items(): setattr(existing, key, value)
        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        return False
    finally:
        session.close()


def record_prediction(*, source: str, model_name: str, prediction, confidence, features: dict, dataset_id=None, device_id=None) -> str | None:
    session = optional_session()
    if session is None:
        return None
    try:
        record_id = str(uuid4())
        session.add(Prediction(
            id=record_id, source=source, model_name=model_name, prediction=str(prediction), confidence=confidence,
            input_features=features, dataset_id=dataset_id, device_id=device_id,
        ))
        session.commit()
        return record_id
    except SQLAlchemyError:
        session.rollback()
        return None
    finally:
        session.close()


def record_model_run(model_name: str, dataset_id: str | None, metrics: dict) -> None:
    session = optional_session()
    if session is None: return
    try:
        session.add(ModelRun(id=str(uuid4()), model_name=model_name, dataset_id=dataset_id, metrics=metrics))
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    finally:
        session.close()


def record_regression_run(algorithm: str, dataset_id: str, target: str, metrics: dict) -> None:
    session = optional_session()
    if session is None: return
    try:
        session.add(RegressionRun(id=str(uuid4()), algorithm=algorithm, dataset_id=dataset_id, target=target, metrics=metrics))
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    finally:
        session.close()


def record_clustering_run(algorithm: str, dataset_id: str, count: int, metrics: dict) -> None:
    session = optional_session()
    if session is None: return
    try:
        session.add(ClusteringRun(id=str(uuid4()), algorithm=algorithm, dataset_id=dataset_id, cluster_count=count, metrics=metrics))
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    finally:
        session.close()


def fetch_predictions(limit: int = 100) -> list[dict]:
    session = optional_session()
    if session is None: return []
    try:
        rows = session.scalars(select(Prediction).order_by(desc(Prediction.timestamp)).limit(limit)).all()
        return [{"id": r.id, "source": r.source, "dataset_id": r.dataset_id, "device_id": r.device_id,
                 "timestamp": r.timestamp.isoformat(), "model_name": r.model_name,
                 "prediction": r.prediction, "confidence": r.confidence, "features": r.input_features} for r in rows]
    finally:
        session.close()


def record_pca_run(dataset_id: str, component_count: int, metrics: dict) -> None:
    session = optional_session()
    if session is None: return
    try:
        session.add(PCARun(id=str(uuid4()), dataset_id=dataset_id, component_count=component_count, metrics=metrics))
        session.commit()
    except SQLAlchemyError:
        session.rollback()
    finally:
        session.close()


def record_predictions_bulk(*, source: str, model_name: str, values: list[dict], dataset_id=None, device_id=None) -> int:
    if not values: return 0
    session = optional_session()
    if session is None: return 0
    try:
        rows = [Prediction(id=str(uuid4()), source=source, model_name=model_name, prediction=str(item["prediction"]),
                           confidence=item.get("confidence"), input_features=item.get("features", {}),
                           dataset_id=dataset_id, device_id=device_id) for item in values]
        session.add_all(rows)
        session.commit()
        return len(rows)
    except SQLAlchemyError:
        session.rollback()
        return 0
    finally:
        session.close()
