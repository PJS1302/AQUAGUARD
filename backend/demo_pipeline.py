from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class DemoWaterPreprocessor:
    pipeline_name = "Demo Water Quality Pipeline"
    preprocessing_steps = [
        "coerce numeric values",
        "median impute missing values",
    ]
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
    fitted_medians: dict[str, float] = {
        "ph": 7.0,
        "Hardness": 240.0,
        "Solids": 17563.0,
        "Chloramines": 4.7,
        "Sulfate": 297.0,
        "Conductivity": 550.0,
        "Organic_carbon": 13.7,
        "Trihalomethanes": 63.0,
        "Turbidity": 4.6,
    }

    def fit(self, X: Any = None, y: Any = None) -> DemoWaterPreprocessor:
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.transform(X)

    def preprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataset copy using the fitted pipeline steps, preserving index."""
        out = frame.copy(deep=True)
        for feature in self.input_features:
            if feature in out.columns:
                out[feature] = pd.to_numeric(out[feature], errors="coerce")
                median_val = self.fitted_medians.get(feature)
                if median_val is not None:
                    out[feature] = out[feature].fillna(median_val)
        return out

    def transform_features(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        """Production feature extraction, ordering, numeric conversion, and fitted imputation."""
        if not isinstance(raw_frame, pd.DataFrame):
            raw_frame = pd.DataFrame(raw_frame, columns=self.input_features)
        missing = [name for name in self.input_features if name not in raw_frame.columns]
        if missing:
            raise ValueError(f"Input is missing required model features: {missing}")
        out = raw_frame.loc[:, self.input_features].copy(deep=True)
        for feature in self.input_features:
            out[feature] = pd.to_numeric(out[feature], errors="coerce")
            median_val = self.fitted_medians.get(feature)
            if median_val is not None:
                out[feature] = out[feature].fillna(median_val)
        return out

    def transform(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        return self.transform_features(raw_frame)


class DemoWaterModel:
    def __init__(self, estimator: LogisticRegression, preprocessor: DemoWaterPreprocessor | None = None):
        self.estimator = estimator
        self.preprocessor = preprocessor or DemoWaterPreprocessor()
        self.feature_names_in_ = list(DemoWaterPreprocessor.input_features)
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))

    def predict(self, X: Any):
        data = self.preprocessor.transform_features(X)
        return self.estimator.predict(data)

    def predict_proba(self, X: Any):
        data = self.preprocessor.transform_features(X)
        return self.estimator.predict_proba(data)
