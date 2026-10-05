"""Small, auditable helpers for lab-side preprocessing.

Production inference does not call these helpers: it uses the exact transformer
saved with the user's trained model. This utility intentionally drops incomplete
rows only when a caller explicitly requests complete cases; it never imputes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from fastapi import HTTPException


def complete_case_copy(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Required columns are missing.", "missing_features": missing})
    copy = frame.loc[:, columns].copy(deep=True)
    copy = copy.replace([np.inf, -np.inf], np.nan)
    before = len(copy)
    copy = copy.dropna(axis=0, how="any").copy()
    return copy, before - len(copy)
