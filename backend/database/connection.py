"""PostgreSQL connection management. There is intentionally no SQLite fallback."""
from __future__ import annotations

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Generator

from fastapi import HTTPException
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

class Base(DeclarativeBase):
    pass

engine = None
SessionLocal = None
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> bool:
    """Create tables when PostgreSQL is configured and reachable."""
    if engine is None:
        logger.warning("DATABASE_URL is not configured; persistent database features are disabled.")
        return False
    try:
        from backend.database import models  # noqa: F401 - register ORM tables
        _migrate_legacy_iot_schema()
        Base.metadata.create_all(bind=engine)
        return True
    except SQLAlchemyError as exc:
        logger.warning("PostgreSQL is not reachable during startup: %s", exc)
        return False


def _migrate_legacy_iot_schema() -> None:
    """Add fields needed by IoT monitoring while preserving the legacy table data."""
    if engine is None:
        return

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "devices" in tables and "device_code" not in {
        column["name"] for column in inspector.get_columns("devices")
    }:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE devices ADD COLUMN device_code VARCHAR(128)"))
            connection.execute(
                text(
                    "UPDATE devices SET device_code = 'legacy-' || id::text "
                    "WHERE device_code IS NULL"
                )
            )
            connection.execute(text("ALTER TABLE devices ALTER COLUMN device_code SET NOT NULL"))
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_devices_device_code "
                    "ON devices (device_code)"
                )
            )

    if "predictions" in tables and "input_features" not in {
        column["name"] for column in inspector.get_columns("predictions")
    }:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE predictions ADD COLUMN input_features JSON"))

    if "datasets" in tables:
        dataset_cols = {column["name"] for column in inspector.get_columns("datasets")}
        with engine.begin() as connection:
            if "is_demo" not in dataset_cols:
                connection.execute(text("ALTER TABLE datasets ADD COLUMN is_demo BOOLEAN DEFAULT FALSE"))
            if "columns" not in dataset_cols:
                connection.execute(text("ALTER TABLE datasets ADD COLUMN columns JSON DEFAULT '[]'::json"))



def database_status() -> dict:
    if engine is None:
        return {"configured": False, "available": False, "message": "Set DATABASE_URL to a PostgreSQL connection string."}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"configured": True, "available": True, "message": "PostgreSQL connected"}
    except SQLAlchemyError as exc:
        logger.warning("Database health check failed: %s", exc)
        return {"configured": True, "available": False, "message": "PostgreSQL is configured but unavailable."}


def optional_session() -> Session | None:
    """Return a session if PostgreSQL is available, otherwise None."""
    if SessionLocal is None:
        return None
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        return session
    except SQLAlchemyError as exc:
        logger.warning("Optional database operation skipped: %s", exc)
        try:
            session.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return None


def require_db() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="PostgreSQL is not configured. Set DATABASE_URL and restart the backend.")
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
        yield session
    except SQLAlchemyError as exc:
        session.rollback()
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable. Check DATABASE_URL and database status.") from exc
    finally:
        session.close()
