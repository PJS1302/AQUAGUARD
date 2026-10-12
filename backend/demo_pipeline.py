from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression


class DemoWaterPreprocessor:
    pipeline_name = "AquaGuard Task-Aware Pipeline"
    preprocessing_steps = ["preserve the source DataFrame; fit task-specific transforms inside the estimator"]
    input_features = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]

    def preprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy(deep=True)

    def transform_features(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.input_features if name not in raw_frame.columns]
        if missing:
            raise ValueError(f"Input is missing required model features: {missing}")
        return raw_frame.loc[:, self.input_features].copy(deep=True)


class DemoWaterModel:
    def __init__(self, estimator: LogisticRegression):
        self.estimator = estimator
        self.feature_names_in_ = list(DemoWaterPreprocessor.input_features)
        self.classes_ = getattr(estimator, "classes_", [0, 1])

    def _prepare(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            data = X.copy(deep=True)
        else:
            data = pd.DataFrame(X, columns=self.feature_names_in_)
        for column in self.feature_names_in_:
            if column not in data.columns:
                raise ValueError(f"Missing feature: {column}")
            data[column] = pd.to_numeric(data[column], errors="coerce")
        return data

    def predict(self, X: Any):
        data = self._prepare(X)
        return self.estimator.predict(data[self.feature_names_in_])

    def predict_proba(self, X: Any):
        data = self._prepare(X)
        return self.estimator.predict_proba(data[self.feature_names_in_])
