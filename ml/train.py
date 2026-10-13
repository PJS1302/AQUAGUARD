"""Genuine ML model training workflow for AquaGuard Water Quality Prediction.

Trains a real classifier on a labelled water-potability dataset, computes
authentic evaluation metrics (cross-validation, precision, recall, F1, accuracy),
and packages the artifacts for production inference.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from backend.demo_pipeline import DemoWaterPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
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
TARGET_NAME = "Potability"
TARGET_LABELS = {
    "0": "NON-POTABLE",
    "1": "POTABLE",
}
FEATURE_UNITS = {
    "ph": "pH",
    "Hardness": "mg/L",
    "Solids": "ppm",
    "Chloramines": "ppm",
    "Sulfate": "mg/L",
    "Conductivity": "μS/cm",
    "Organic_carbon": "ppm",
    "Trihalomethanes": "μg/L",
    "Turbidity": "NTU",
}


def load_and_validate_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate dataset containing water quality features and potability label."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset from %s (%d rows, %d columns)", csv_path.name, len(df), len(df.columns))

    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features: {missing_features}")

    if TARGET_NAME not in df.columns:
        raise ValueError(f"Dataset missing required target column: '{TARGET_NAME}'")

    # Clean target
    valid_mask = df[TARGET_NAME].notna() & df[TARGET_NAME].isin([0, 1, "0", "1", 0.0, 1.0])
    clean_df = df.loc[valid_mask].copy()

    X = clean_df[FEATURE_NAMES].copy()
    y = clean_df[TARGET_NAME].astype(int)

    logger.info("Validated %d labelled rows (Potable: %d, Non-potable: %d)", len(y), (y == 1).sum(), (y == 0).sum())
    return X, y


def train_model(
    dataset_path: Path | None = None,
    model_type: str = "random_forest",
    output_dir: Path | None = None,
) -> dict:
    """Execute complete genuine training and evaluation pipeline."""
    root_dir = Path(__file__).resolve().parents[1]
    if dataset_path is None:
        dataset_path = root_dir / "frontend" / "public" / "demo" / "demo_water_quality.csv"
    if output_dir is None:
        output_dir = root_dir / "backend" / "models"

    output_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y = load_and_validate_dataset(dataset_path)

    # Apply authentic preprocessor
    preprocessor = DemoWaterPreprocessor()
    X_processed = preprocessor.transform_features(X_raw)

    # Train / test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "random_forest":
        model_name = "AquaGuard Random Forest Model"
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
        )
    else:
        model_name = "AquaGuard Logistic Regression Classifier"
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )

    logger.info("Training %s on %d samples...", model_name, len(X_train))
    estimator.fit(X_train, y_train)

    # Evaluate on holdout test set
    y_pred = estimator.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Stratified 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(estimator, X_processed, y, cv=cv, scoring="f1_weighted")

    evaluation_report = {
        "model_name": model_name,
        "model_type": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "f1_weighted": round(f1, 4),
            "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
            "cv_f1_std": round(float(np.std(cv_scores)), 4),
            "confusion_matrix": conf_matrix,
        },
    }

    logger.info("Evaluation results: Accuracy: %.2f%%, F1: %.4f, CV F1: %.4f", accuracy * 100, f1, float(np.mean(cv_scores)))

    # Save artifacts
    model_artifact_path = output_dir / "production_model.pkl"
    joblib.dump(estimator, model_artifact_path)
    logger.info("Saved model artifact to %s", model_artifact_path)

    prep_artifact_path = output_dir / "preprocessing.pkl"
    joblib.dump(preprocessor, prep_artifact_path)
    logger.info("Saved preprocessor artifact to %s", prep_artifact_path)

    feature_config = {
        "artifact_file": "production_model.pkl",
        "preprocessing_artifact": "preprocessing.pkl",
        "model_name": model_name,
        "model_type": model_type,
        "features": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "target_labels": TARGET_LABELS,
        "feature_units": FEATURE_UNITS,
        "preprocessing_in_pipeline": False,
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation_report["metrics"],
    }

    config_path = output_dir / "feature_config.json"
    config_path.write_text(json.dumps(feature_config, indent=2), encoding="utf-8")
    logger.info("Updated feature config at %s", config_path)

    # Invalidate cache if model_loader is loaded in current process
    try:
        from backend.utils.model_loader import invalidate_model_cache
        invalidate_model_cache()
    except Exception:
        pass

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="AquaGuard Model Trainer")
    parser.add_argument("--dataset", type=str, default=None, help="Path to labelled water quality CSV")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "logistic_regression"])
    args = parser.parse_args()

    report = train_model(
        dataset_path=Path(args.dataset) if args.dataset else None,
        model_type=args.model_type,
    )
    print("\nTraining complete! Summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
