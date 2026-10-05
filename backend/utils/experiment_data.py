from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException

MIN_TRAIN_TEST_ROWS = 12


def require_complete_processed_rows(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    task: str,
    minimum: int = MIN_TRAIN_TEST_ROWS,
) -> pd.DataFrame:
    """Validate processed inputs without dropping or imputing any row."""
    selected = frame.loc[:, columns].copy(deep=True)
    complete = ~selected.isna().any(axis=1)
    complete_count = int(complete.sum())
    incomplete_count = int((~complete).sum())
    if complete_count < minimum:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"{task} was not run: at least {minimum} complete processed rows are "
                    f"required after the user's preprocessing pipeline; only {complete_count} remain."
                ),
                "processed_rows": int(len(selected)),
                "complete_processed_rows": complete_count,
                "incomplete_processed_rows": incomplete_count,
                "minimum_complete_rows": int(minimum),
                "application_dropped_or_imputed_rows": False,
            },
        )
    if incomplete_count:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"{task} was not run: the user's preprocessing output still contains "
                    f"{incomplete_count} incomplete selected row(s). AquaGuard will not "
                    "drop or impute them. Update the original pipeline if that is its intended behavior."
                ),
                "processed_rows": int(len(selected)),
                "complete_processed_rows": complete_count,
                "incomplete_processed_rows": incomplete_count,
                "minimum_complete_rows": int(minimum),
                "application_dropped_or_imputed_rows": False,
            },
        )
    return selected


def require_finite_numeric(frame: pd.DataFrame, columns: list[str], *, task: str) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{task} requires the user's preprocessing output for '{column}' to be numeric. "
                    "AquaGuard will not coerce, encode, or scale this column."
                ),
            )
        try:
            values = frame[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Processed column '{column}' is not numeric.") from exc
        if not np.isfinite(values).all():
            raise HTTPException(
                status_code=422,
                detail=f"Processed column '{column}' contains a non-finite value; AquaGuard will not replace it.",
            )
