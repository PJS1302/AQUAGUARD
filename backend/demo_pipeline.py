from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression


class DemoWaterPreprocessor:
    pipeline_name = "Demo Water Quality Pipeline"
    preprocessing_steps = ["coerce numeric values", "median impute missing values"]
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
        candidate = frame.copy(deep=True)
        for column in self.input_features + ["Potability"]:
            if column in candidate.columns:
                candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
        for column in self.input_features + ["Potability"]:
            if column in candidate.columns:
                candidate[column] = candidate[column].fillna(candidate[column].median())
        return candidate

    def transform_features(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.preprocess_frame(raw_frame)
        return transformed.loc[:, self.input_features].copy(deep=True)


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
