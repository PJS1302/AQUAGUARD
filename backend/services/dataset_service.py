from __future__ import annotations
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from uuid import uuid4, UUID

import numpy as np
import pandas as pd
from fastapi import HTTPException
from backend.config import MAX_UPLOAD_BYTES, PROCESSED_DIR, UPLOAD_DIR, PROJECT_DIR
from backend.utils.dataset_validator import json_safe, validate_csv_bytes
from backend.utils.experiment_data import profile_frame


def _safe_name(filename: str) -> str:
    base = Path(filename or "dataset.csv").name.replace("\x00", "")
    cleaned = "".join(ch for ch in base if ch.isalnum() or ch in "._- ").strip()
    if not cleaned:
        cleaned = "dataset.csv"
    return cleaned[:180]


def inspect_dataframe(frame: pd.DataFrame) -> dict:
    profiles = profile_frame(frame)
    roles = {item["name"]: item for item in profiles}
    empty_columns = [column for column, item in roles.items() if item["role"] == "empty"]
    numeric = [column for column, item in roles.items() if item["role"] == "numerical"]
    categorical = [column for column, item in roles.items() if item["role"] in {"categorical", "boolean"}]
    identifiers = [column for column, item in roles.items() if item["role"] == "identifier"]
    dates = [column for column, item in roles.items() if item["role"] == "datetime"]
    complete_numeric = [
        column
        for column in numeric
        if roles[str(column)]["missing_values"] == 0
        and roles[str(column)]["invalid_values"] == 0
        and roles[str(column)]["infinite_values"] == 0
    ]
    classification_targets = []
    for column in frame.columns:
        if column in empty_columns or column in identifiers:
            continue
        values = frame[column].dropna()
        if values.dtype == object or isinstance(values.dtype, pd.StringDtype):
            text = values.astype("string").str.strip()
            values = values[~text.str.lower().isin({"", "na", "n/a", "nan", "null", "none", "nat"})]
        counts = values.value_counts(dropna=True)
        class_count = len(counts)
        numeric_target = pd.api.types.is_numeric_dtype(values)
        looks_continuous = numeric_target and class_count / max(1, len(values)) >= 0.2
        if class_count >= 2 and int(counts.min()) >= 2 and not looks_continuous:
            classification_targets.append((column, class_count, int(counts.min())))
    classification_targets.sort(key=lambda item: (item[1], -item[2], str(item[0])))
    missing = {str(c): int(v) for c, v in frame.isna().sum().items() if int(v) > 0}
    dtype_map = {str(c): str(t) for c, t in frame.dtypes.items()}
    duplicate_count = int(frame.duplicated().sum())
    stats = {}
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if len(values):
            stats[str(column)] = {
                "min": json_safe(np.min(values)),
                "max": json_safe(np.max(values)),
                "mean": json_safe(np.mean(values)),
                "std": json_safe(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    preview = frame.head(8).replace({np.nan: None}).to_dict(orient="records")
    return {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "columns": [str(c) for c in frame.columns],
        "numeric_columns": [str(c) for c in numeric],
        "complete_numeric_columns": [str(c) for c in complete_numeric],
        "classification_target_candidates": [str(item[0]) for item in classification_targets],
        "categorical_columns": [str(c) for c in categorical],
        "boolean_columns": [str(c) for c, item in roles.items() if item["role"] == "boolean"],
        "datetime_columns": [str(c) for c in dates],
        "identifier_columns": [str(c) for c in identifiers],
        "empty_columns": [str(c) for c in empty_columns],
        "feature_columns": [str(c) for c in frame.columns if c not in identifiers and c not in empty_columns],
        "column_profiles": profiles,
        "dtypes": dtype_map,
        "missing_value_summary": missing,
        "missing_total": int(frame.isna().sum().sum()),
        "duplicate_rows": duplicate_count,
        "numeric_summary": stats,
        "preview": json_safe(preview),
    }


def _is_bundled_demo(raw: bytes) -> bool:
    demo_path = PROJECT_DIR / "frontend" / "public" / "demo" / "demo_water_quality.csv"
    try:
        expected = hashlib.sha256(demo_path.read_bytes()).digest()
        return hashlib.sha256(raw).digest() == expected
    except OSError:
        return False


def save_uploaded_bytes(raw: bytes, filename: str, is_demo: bool = False) -> dict:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB file-size limit.")
    safe_name = _safe_name(filename)
    is_demo = bool(is_demo or _is_bundled_demo(raw))
    frame = validate_csv_bytes(raw, safe_name)
    dataset_id = str(uuid4())
    folder = UPLOAD_DIR / dataset_id
    folder.mkdir(parents=True, exist_ok=False)
    original_path = folder / safe_name
    # Save the original upload bytes exactly as received; all analysis uses a separate DataFrame copy.
    original_path.write_bytes(raw)
    summary = inspect_dataframe(frame)
    sidecar = {
        "id": dataset_id,
        "file_name": safe_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "is_demo": bool(is_demo),
        "compatibility_status": "inspection_only",
        **summary,
    }
    (folder / "metadata.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**sidecar, "_path": str(original_path)}


def _validated_id(dataset_id: str) -> str:
    try:
        return str(UUID(dataset_id))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid dataset id.") from exc


def get_dataset_path(dataset_id: str) -> Path:
    clean_id = _validated_id(dataset_id)
    folder = (UPLOAD_DIR / clean_id).resolve()
    if UPLOAD_DIR.resolve() not in folder.parents:
        raise HTTPException(status_code=400, detail="Invalid dataset path.")
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Dataset file was not found. Upload the CSV again after a backend restart if needed.")
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".csv"]
    if len(files) != 1:
        raise HTTPException(status_code=404, detail="Original CSV was not found for this dataset.")
    return files[0]


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    path = get_dataset_path(dataset_id)
    metadata_path = path.parent / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        metadata = {"id": dataset_id, "file_name": path.name, "is_demo": False}
    try:
        original = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The preserved CSV could not be read.") from exc
    return original, metadata


def list_local_datasets() -> list[dict]:
    result = []
    for folder in UPLOAD_DIR.iterdir():
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                result.append({k: v for k, v in meta.items() if k != "preview"})
            except Exception:
                continue
    return sorted(result, key=lambda item: item.get("uploaded_at", ""), reverse=True)


def save_processed_copy(frame: pd.DataFrame) -> str:
    process_id = str(uuid4())
    path = PROCESSED_DIR / f"{process_id}.csv"
    # This is a new result file, never a write-back to the original upload.
    frame.to_csv(path, index=False)
    return process_id


def get_processed_path(process_id: str) -> Path:
    clean = _validated_id(process_id)
    path = (PROCESSED_DIR / f"{clean}.csv").resolve()
    if PROCESSED_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Processed export was not found.")
    return path
