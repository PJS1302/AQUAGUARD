from __future__ import annotations
import io
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.config import MAX_COLUMNS, MAX_ROWS

ALLOWED_SUFFIXES = {".csv"}


def validate_csv_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    if Path(filename or "").suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only CSV files are supported.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, [])
        if header:
            normalized = [str(value).strip() for value in header]
            if any(not name for name in normalized):
                raise HTTPException(status_code=400, detail="Column names cannot be blank.")
            duplicates = sorted({name for name in normalized if normalized.count(name) > 1 and name})
            if duplicates:
                raise HTTPException(status_code=400, detail=f"Duplicate column names are not supported: {duplicates}")
        frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise HTTPException(status_code=400, detail="The CSV must contain at least one data row.")
    if len(frame) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"CSV exceeds the {MAX_ROWS:,}-row limit.")
    if len(frame.columns) > MAX_COLUMNS:
        raise HTTPException(status_code=413, detail=f"CSV exceeds the {MAX_COLUMNS}-column limit.")
    if frame.columns.duplicated().any():
        dupes = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise HTTPException(status_code=400, detail=f"Duplicate column names are not supported: {dupes}")
    if any(not str(column).strip() for column in frame.columns):
        raise HTTPException(status_code=400, detail="Column names cannot be blank.")


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
