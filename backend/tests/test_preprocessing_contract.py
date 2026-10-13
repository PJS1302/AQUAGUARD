from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from backend.routes import dataset as dataset_route
from backend.services import user_preprocessing_service
from backend.services.classification_service import (
    _prepare as prepare_classification,
    compare_classifiers,
    run_classification,
)
from backend.services.clustering_service import run_kmeans
from backend.services import dataset_service
from backend.services.dataset_service import inspect_dataframe
from backend.services.pca_service import run_pca
from backend.services.regression_service import prepare_regression_data, run_regression
from backend.utils.dataset_validator import validate_csv_bytes
from backend.utils.experiment_data import make_preprocessing_pipeline, normalize_feature_frame
from backend.utils import model_loader


class DropMissingPipeline:
    pipeline_name = "test fixture only"
    preprocessing_steps = ["explicit fixture rule"]

    def preprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.dropna(subset=["x"]).copy()

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
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

    def test_user_processing_copy_keeps_source_unchanged(self):
        source = pd.DataFrame({"x": [1.0, None, 3.0], "y": [2.0, 4.0, 6.0]})
        preserved = source.copy(deep=True)
        with patch.object(
            user_preprocessing_service,
            "get_user_pipeline",
            return_value=(DropMissingPipeline(), {}, self._status()),
        ):
            processed, report = user_preprocessing_service.apply_user_preprocessing(source)
        self.assertEqual(len(processed), 2)
        self.assertEqual(report["rows_removed_by_pipeline"], 1)
        assert_frame_equal(source, preserved)

    def test_small_regression_imputes_features_without_dropping_rows(self):
        frame = pd.DataFrame({"x": [1.0, None, 3.0], "target": [2.0, 4.0, 7.0]})
        preserved = frame.copy(deep=True)

        X, y, diagnostics = prepare_regression_data(frame, "target", ["x"])
        result = run_regression(
            frame,
            dataset_id="test",
            algorithm="linear",
            target="target",
            features=["x"],
            folds=5,
        )

        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(diagnostics["feature_missing_values"], {"x": 1})
        self.assertEqual(result["complete_rows"], 3)
        self.assertIsNone(result["metrics"]["r2"])
        self.assertEqual(result["cross_validation"]["folds"], 0)
        assert_frame_equal(frame, preserved)

    def test_two_class_four_row_holdout_has_no_universal_row_minimum(self):
        frame = pd.DataFrame({"x": [1, 2, 3, 4], "label": ["a", "a", "b", "b"]})
        with (
            patch("backend.services.classification_service._save_experiment", return_value="test"),
            patch("backend.services.classification_service.record_model_run"),
        ):
            result = run_classification(
                frame,
                dataset_id="test",
                dataset_metadata={"is_demo": True},
                target="label",
                features=["x"],
                algorithm="decision-tree",
                folds=5,
            )
        self.assertEqual(result["complete_rows"], 4)
        self.assertEqual(len(result["actual"]), 2)
        self.assertIn("accuracy", result["metrics"])

    def test_classification_handles_mixed_types_dates_missing_and_target_exclusions(self):
        frame = pd.DataFrame(
            {
                "reading": [1, 2, None, 4, "invalid", 6, 7, 8, 9, 10, 11, 12],
                "station": ["north", "south"] * 6,
                "enabled": [True, False] * 6,
                "sample_time": pd.date_range("2025-01-01", periods=12).astype(str),
                "label": ["safe", "unsafe"] * 6,
            }
        )
        frame.loc[1, "label"] = None
        preserved = frame.copy(deep=True)
        X, y, numeric, categorical, diagnostics = prepare_classification(
            frame, "label", ["reading", "station", "enabled", "sample_time"]
        )

        self.assertEqual(len(X), 11)
        self.assertEqual(len(y), 11)
        self.assertIn("reading", numeric)
        self.assertIn("station", categorical)
        self.assertEqual(diagnostics["feature_roles"]["enabled"], "boolean")
        self.assertEqual(diagnostics["feature_roles"]["sample_time"], "datetime")
        self.assertEqual(diagnostics["target"]["invalid_or_missing_rows_excluded"], 1)
        assert_frame_equal(frame, preserved)

    def test_training_fitted_pipeline_reuses_transform_for_unseen_categories_and_missing_values(self):
        train = pd.DataFrame({"amount": [1.0, None, 3.0, 4.0], "kind": ["a", "a", "b", "b"]})
        test = pd.DataFrame({"amount": [1_000_000.0, None], "kind": ["new", "b"]})
        normalized, diagnostics = normalize_feature_frame(train, ["amount", "kind"])
        roles = {item["name"]: item["role"] for item in diagnostics["columns"]}
        pipeline = make_preprocessing_pipeline(train, LogisticRegression(), roles)
        pipeline.fit(train, [0, 0, 1, 1])
        predictions = pipeline.predict(test)
        median = pipeline.named_steps["preprocessing"].named_transformers_["numeric"].statistics_[0]

        self.assertEqual(float(median), 3.0)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(normalized), 4)

    def test_model_comparison_runs_compatible_classifiers_on_small_mixed_data(self):
        frame = pd.DataFrame(
            {
                "reading": [float(i) if i != 3 else np.nan for i in range(20)],
                "kind": ["north", "south"] * 10,
                "label": ["safe", "unsafe"] * 10,
            }
        )
        with (
            patch("backend.services.classification_service._save_experiment", return_value="test"),
            patch("backend.services.classification_service.record_model_run"),
        ):
            result = compare_classifiers(
                frame,
                dataset_id="test",
                dataset_metadata={"is_demo": True},
                target="label",
                features=["reading", "kind"],
                folds=5,
                k=5,
                n_estimators=5,
                learning_rate=0.8,
            )
        self.assertEqual(len(result["results"]), 7)
        self.assertFalse(result["errors"])
        self.assertTrue(all(item["metrics"]["accuracy"] is not None for item in result["results"]))

    def test_pca_and_clustering_apply_reported_unsupervised_imputation(self):
        frame = pd.DataFrame(
            {
                "x": [1.0, None, 3.0, 4.0, 5.0],
                "group": ["a", "b", "a", "b", "a"],
            }
        )
        with (
            patch("backend.services.pca_service.record_pca_run"),
            patch("backend.services.clustering_service.record_clustering_run"),
        ):
            pca = run_pca(frame, ["x"], n_components=1)
            clusters = run_kmeans(frame, "test", ["x", "group"], n_clusters=2)
        self.assertEqual(pca["complete_rows"], 5)
        self.assertEqual(pca["component_count"], 1)
        self.assertEqual(len(clusters["labels"]), 5)
        self.assertIn("group_a", clusters["features"])
        self.assertEqual(clusters["feature_preprocessing"]["application_dropped_rows"], 0)

    def test_duplicate_csv_headers_are_rejected_before_pandas_renames_them(self):
        with self.assertRaises(HTTPException) as caught:
            validate_csv_bytes(b"value,value\n1,2\n", "duplicate.csv")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("Duplicate column names", caught.exception.detail)

    def test_integrated_fitted_model_handles_single_and_multiple_predictions(self):
        train = pd.DataFrame(
            {"reading": [1.0, 2.0, None, 4.0, 5.0, 6.0], "kind": ["a", "a", "b", "b", "a", "b"]}
        )
        labels = pd.Series([0, 0, 0, 1, 1, 1])
        _, diagnostics = normalize_feature_frame(train, list(train.columns))
        roles = {item["name"]: item["role"] for item in diagnostics["columns"]}
        pipeline = make_preprocessing_pipeline(train, LogisticRegression(), roles)
        pipeline.fit(train, labels)
        bundle = {
            "available": True,
            "model": pipeline,
            "preprocessing": None,
            "scaler": None,
            "integrated": True,
            "features": ["reading", "kind"],
            "model_name": "fitted test pipeline",
            "config": {"artifact_file": "test.joblib"},
        }
        with patch.object(model_loader, "get_model_bundle", return_value=bundle):
            one = model_loader.predict_one({"reading": None, "kind": "unseen"})
            many = model_loader.predict_frame(
                pd.DataFrame({"kind": ["a", "new"], "reading": [2.5, None]})
            )
        self.assertIn(one["prediction"], [0, 1])
        self.assertEqual(len(many["predictions"]), 2)
        self.assertEqual(many["model_name"], "fitted test pipeline")

    def test_unfitted_production_model_reports_missing_preprocessor_without_fitting_on_request(self):
        model = LogisticRegression().fit([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1])
        bundle = {
            "available": True,
            "model": model,
            "preprocessing": None,
            "scaler": None,
            "integrated": False,
            "features": ["reading"],
            "model_name": "selected production model",
            "config": {"artifact_file": "production_model.pkl"},
        }
        with patch.object(model_loader, "get_model_bundle", return_value=bundle):
            with self.assertRaises(HTTPException) as caught:
                model_loader.predict_frame(pd.DataFrame({"reading": [None]}))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("has no fitted preprocessing object", caught.exception.detail["message"])
        self.assertIn("will not fit an imputer", caught.exception.detail["required_action"])

    def test_inspection_reports_types_and_excludes_empty_and_identifier_defaults(self):
        frame = pd.DataFrame(
            {
                "Longitude": [1.0, 2.0, 3.0, 4.0],
                "Chemical Oxygen Demand (mg/L)": [None, None, None, None],
                "Partially observed": [4.0, None, 6.0, 8.0],
                "District": ["North", "North", "South", "South"],
                "sample_time": pd.date_range("2024-01-01", periods=4).astype(str),
                "station_id": [1, 2, 3, 4],
            }
        )
        preserved = frame.copy(deep=True)
        report = inspect_dataframe(frame)

        self.assertEqual(report["numeric_columns"], ["Longitude", "Partially observed"])
        self.assertEqual(report["complete_numeric_columns"], ["Longitude"])
        self.assertNotIn("Chemical Oxygen Demand (mg/L)", report["feature_columns"])
        self.assertNotIn("station_id", report["feature_columns"])
        self.assertIn("sample_time", report["datetime_columns"])
        self.assertIn("District", report["classification_target_candidates"])
        self.assertEqual(report["missing_value_summary"]["Chemical Oxygen Demand (mg/L)"], 4)
        assert_frame_equal(frame, preserved)

    def test_upload_endpoint_accepts_numeric_columns_with_mixed_values(self):
        raw = (
            b"measurement,group\n"
            b"1.5,north\n"
            b"7.0,north\n"
            b"3.0,south\n"
            b"Clear,south\n"
            b"5.5,north\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(dataset_service, "UPLOAD_DIR", Path(temp_dir)),
                patch.object(dataset_route, "persist_dataset", return_value=True),
            ):
                uploaded = asyncio.run(
                    dataset_route.upload_dataset(
                        dataset_route.UploadFile(
                            filename="mixed-measurement.csv",
                            file=io.BytesIO(raw),
                        ),
                        is_demo=False,
                    )
                )
                preserved_path = dataset_service.get_dataset_path(uploaded["id"])
                self.assertEqual(preserved_path.read_bytes(), raw)

        self.assertEqual(uploaded["row_count"], 5)
        self.assertEqual(uploaded["column_count"], 2)
        self.assertEqual(uploaded["numeric_summary"]["measurement"]["min"], 1.5)
        self.assertEqual(uploaded["numeric_summary"]["measurement"]["max"], 7.0)
        self.assertEqual(uploaded["numeric_summary"]["measurement"]["mean"], 4.25)
        self.assertEqual(uploaded["column_profiles"][0]["invalid_values"], 1)
        self.assertTrue(uploaded["postgres_persisted"])


if __name__ == "__main__":
    unittest.main()
