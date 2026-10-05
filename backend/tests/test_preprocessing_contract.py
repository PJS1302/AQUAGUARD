from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from pandas.testing import assert_frame_equal

from backend.services import user_preprocessing_service
from backend.services.classification_service import _prepare as prepare_classification
from backend.services.regression_service import prepare_regression_data


class DropMissingPipeline:
    pipeline_name = "test fixture only"
    preprocessing_steps = ["explicit fixture rule"]

    def preprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.dropna(subset=["x"]).copy()

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


class IdentityPipeline:
    pipeline_name = "identity test fixture"

    def preprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


class PreprocessingContractTests(unittest.TestCase):
    def _status(self) -> dict:
        return {
            "available": True,
            "pipeline_name": "test fixture only",
            "steps": ["explicit fixture rule"],
            "artifact_file": "test-only",
            "artifact_sha256": "test-sha256",
        }

    def test_pipeline_only_drops_rows_in_processing_copy(self):
        source = pd.DataFrame({
            "x": [float(value) for value in range(12)] + [None],
            "y": [float(value * 2) for value in range(13)],
            "label": ["a", "b"] * 6 + ["a"],
        })
        preserved = source.copy(deep=True)
        with patch.object(user_preprocessing_service, "get_user_pipeline", return_value=(DropMissingPipeline(), {}, self._status())):
            processed, report = user_preprocessing_service.apply_user_preprocessing(source)
        self.assertEqual(len(processed), 12)
        self.assertEqual(report["rows_removed_by_pipeline"], 1)
        assert_frame_equal(source, preserved)
        self.assertEqual(list(processed.index), list(range(12)))

    def test_supervised_experiment_requires_twelve_complete_processed_rows(self):
        frame = pd.DataFrame({
            "x": [float(value) for value in range(12)],
            "y": [float(value * 2) for value in range(12)],
            "label": ["a", "b"] * 6,
        })
        X, target, _, _ = prepare_classification(frame, "label", ["x", "y"])
        self.assertEqual(len(X), 12)
        self.assertEqual(len(target), 12)
        X_reg, y_reg, _ = prepare_regression_data(frame, "y", ["x"])
        self.assertEqual(len(X_reg), 12)
        self.assertEqual(len(y_reg), 12)

        with self.assertRaises(HTTPException) as caught:
            prepare_classification(frame.iloc[:11].copy(), "label", ["x", "y"])
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail["minimum_complete_rows"], 12)
        self.assertEqual(caught.exception.detail["complete_processed_rows"], 11)

    def test_application_does_not_drop_or_impute_rows_left_incomplete_by_pipeline(self):
        frame = pd.DataFrame({
            "x": [float(value) for value in range(12)],
            "y": [float(value * 2) for value in range(12)],
        })
        frame.loc[0, "x"] = None
        with patch.object(user_preprocessing_service, "get_user_pipeline", return_value=(IdentityPipeline(), {}, self._status())):
            processed, report = user_preprocessing_service.apply_user_preprocessing(frame)
        self.assertEqual(len(processed), 12)
        self.assertEqual(report["rows_removed_by_pipeline"], 0)
        self.assertEqual(report["missing_cells_after"], 1)
        with self.assertRaises(HTTPException) as caught:
            prepare_regression_data(processed, "y", ["x"])
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIs(caught.exception.detail["application_dropped_or_imputed_rows"], False)
        self.assertEqual(len(processed), 12)


if __name__ == "__main__":
    unittest.main()
