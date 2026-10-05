from __future__ import annotations
import numpy as np
import pandas as pd


def data_quality_report(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include=[np.number])
    outliers = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        outliers[str(column)] = {"count": count, "method": "1.5 × IQR review flag; not an automatic removal"}
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_by_column": {str(c): int(v) for c, v in frame.isna().sum().items() if int(v) > 0},
        "duplicate_rows": int(frame.duplicated().sum()),
        "data_types": {str(c): str(t) for c, t in frame.dtypes.items()},
        "outlier_review": outliers,
        "original_preserved": True,
    }
