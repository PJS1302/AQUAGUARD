"""Genuine model evaluation workflow with SHAP explainability verification."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from backend.utils.model_loader import get_model_bundle
from backend.services.shap_service import explain_prediction, get_global_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_production_model() -> dict:
    """Evaluate the currently installed production model and generate SHAP summary."""
    bundle = get_model_bundle(refresh=True)
    if not bundle.get("available"):
        logger.error("No production model is installed: %s", bundle.get("reason"))
        return {"status": "unavailable", "reason": bundle.get("reason")}

    model = bundle["model"]
    preprocessor = bundle.get("preprocessing")
    features = bundle["features"]
    config = bundle.get("config", {})

    logger.info("Evaluating installed model: %s", bundle["model_name"])

    # Load test dataset
    demo_path = Path(__file__).resolve().parents[1] / "frontend" / "public" / "demo" / "demo_water_quality.csv"
    if not demo_path.is_file():
        return {"status": "error", "message": "Demo evaluation dataset not found."}

    df = pd.read_csv(demo_path)
    X_raw = df[features]
    y_true = df["Potability"].astype(int)

    if preprocessor is not None and hasattr(preprocessor, "transform_features"):
        X_trans = preprocessor.transform_features(X_raw)
    else:
        X_trans = X_raw

    y_pred = model.predict(X_trans)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["NON-POTABLE", "POTABLE"], output_dict=True)

    # Compute global SHAP feature importance
    shap_summary = get_global_feature_importance(model, preprocessor, features, units=config.get("feature_units"))

    # Compute sample local explanation for row 0
    sample_raw = {col: float(X_raw.iloc[0][col]) for col in features}
    sample_trans = X_trans.iloc[[0]]
    sample_shap = explain_prediction(
        model=model,
        preprocessor=preprocessor,
        raw_features=sample_raw,
        transformed_frame=sample_trans,
        features=features,
        units=config.get("feature_units"),
        target_labels=config.get("target_labels"),
    )

    results = {
        "model_name": bundle["model_name"],
        "model_type": type(model).__name__,
        "evaluated_samples": len(df),
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "global_shap_importance": shap_summary,
        "sample_local_shap": sample_shap,
    }
    return results


def main():
    results = evaluate_production_model()
    print("\nAquaGuard Production Model Evaluation Report:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
