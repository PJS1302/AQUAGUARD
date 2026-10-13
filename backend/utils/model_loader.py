from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.config import MODELS_DIR
from backend.utils.dataset_validator import json_safe

logger = logging.getLogger(__name__)
_CACHE: dict[str, Any] | None = None
_CACHE_SIGNATURE = None

def _signature():
    names = ("aquaguard_pipeline.pkl", "production_model.pkl", "preprocessing.pkl", "scaler.pkl", "user_preprocessing_pipeline.joblib", "feature_config.json")
    values = []
    for name in names:
        path = MODELS_DIR / name
        try:
            stat = path.stat()
            values.append((name, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            values.append((name, None, None))
    try:
        active = json.loads((MODELS_DIR / "feature_config.json").read_text(encoding="utf-8"))
        active_name = str(active.get("artifact_file", ""))
        if active_name and Path(active_name).name == active_name:
            stat = (MODELS_DIR / active_name).stat()
            values.append((active_name, stat.st_mtime_ns, stat.st_size))
        prep_name = str(active.get("preprocessing_artifact", ""))
        if prep_name and Path(prep_name).name == prep_name:
            stat = (MODELS_DIR / prep_name).stat()
            values.append((prep_name, stat.st_mtime_ns, stat.st_size))
    except (FileNotFoundError, ValueError, OSError):
        pass
    return tuple(values)



def _load_bundle() -> dict:
    config_path = MODELS_DIR / "feature_config.json"
    pipeline_path = MODELS_DIR / "aquaguard_pipeline.pkl"
    model_path = MODELS_DIR / "production_model.pkl"
    config: dict = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"available": False, "reason": f"feature_config.json is invalid: {exc}"}

    integrated = False
    explicit_artifact = config.get("artifact_file")
    if explicit_artifact:
        artifact_name = str(explicit_artifact)
        if Path(artifact_name).name != artifact_name or Path(artifact_name).suffix.lower() != ".pkl":
            return {"available": False, "reason": "feature_config.json 'artifact_file' must be a local .pkl filename inside backend/models/."}
        selected_path = MODELS_DIR / artifact_name
        if not selected_path.is_file():
            return {"available": False, "reason": f"Configured model artifact '{artifact_name}' was not found."}
        try:
            artifact = joblib.load(selected_path)
            integrated = artifact_name == "aquaguard_pipeline.pkl" or isinstance(artifact, dict) and ("pipeline" in artifact or "model" in artifact)
            if isinstance(artifact, dict):
                model = artifact.get("pipeline") or artifact.get("model")
                config = {**artifact.get("feature_config", {}), **config}
                if artifact.get("features") and not config.get("features"):
                    config["features"] = artifact["features"]
                model_name = artifact.get("model_name") or config.get("model_name") or Path(artifact_name).stem
            else:
                model = artifact
                model_name = config.get("model_name") or Path(artifact_name).stem
        except Exception as exc:
            return {"available": False, "reason": f"Could not load configured artifact {artifact_name}: {exc}"}
    elif pipeline_path.exists():
        try:
            artifact = joblib.load(pipeline_path)
            integrated = True
            if isinstance(artifact, dict):
                model = artifact.get("pipeline") or artifact.get("model")
                config = {**artifact.get("feature_config", {}), **config}
                if artifact.get("features") and not config.get("features"):
                    config["features"] = artifact["features"]
                model_name = artifact.get("model_name") or config.get("model_name") or "Integrated pipeline"
            else:
                model = artifact
                model_name = config.get("model_name") or "Integrated pipeline"
        except Exception as exc:
            return {"available": False, "reason": f"Could not load aquaguard_pipeline.pkl: {exc}"}
    elif model_path.exists():
        try:
            model = joblib.load(model_path)
            model_name = config.get("model_name") or type(model).__name__
        except Exception as exc:
            return {"available": False, "reason": f"Could not load production_model.pkl: {exc}"}
    else:
        return {"available": False, "reason": "No production model artifact is installed."}

    features = config.get("features")
    if not features and hasattr(model, "feature_names_in_"):
        features = [str(x) for x in model.feature_names_in_]
    if not features:
        return {"available": False, "reason": "Model exists, but feature_config.json must declare the exact ordered feature list."}
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        return {"available": False, "reason": "feature_config.json 'features' must be an ordered list of column names."}

    integrated = integrated or bool(config.get("preprocessing_in_pipeline", False))
    preprocessing = None
    scaler = None
    if not integrated:
        prep_file = config.get("preprocessing_artifact") or "preprocessing.pkl"
        prep_path = MODELS_DIR / prep_file
        scaler_file = config.get("scaler_artifact") or "scaler.pkl"
        scaler_path = MODELS_DIR / scaler_file
        try:
            if prep_path.exists():
                prep_obj = joblib.load(prep_path)
                if isinstance(prep_obj, dict) and "pipeline" in prep_obj:
                    preprocessing = prep_obj["pipeline"]
                else:
                    preprocessing = prep_obj
            elif (MODELS_DIR / "preprocessing.pkl").exists():
                preprocessing = joblib.load(MODELS_DIR / "preprocessing.pkl")
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
        except Exception as exc:
            return {"available": False, "reason": f"Could not load preprocessing artifact: {exc}"}
    return {
        "available": True,
        "model": model,
        "preprocessing": preprocessing,
        "scaler": scaler,
        "integrated": integrated,
        "features": features,
        "model_name": str(model_name),
        "config": config,
    }


def get_model_bundle(refresh: bool = False) -> dict:
    global _CACHE, _CACHE_SIGNATURE
    signature = _signature()
    if refresh or _CACHE is None or signature != _CACHE_SIGNATURE:
        _CACHE = _load_bundle()
        _CACHE_SIGNATURE = signature
    return _CACHE


def model_status() -> dict:
    bundle = get_model_bundle()
    if not bundle.get("available"):
        return {
            "available": False,
            "reason": bundle.get("reason"),
            "model_name": None,
            "features": [],
            "feature_units": {},
            "target_labels": {},
            "shap_supported": False,
        }
    config = bundle.get("config", {})
    return {
        "available": True,
        "reason": None,
        "model_name": bundle["model_name"],
        "features": bundle["features"],
        "feature_units": config.get("feature_units", {}),
        "target_labels": config.get("target_labels", {"0": "NON-POTABLE", "1": "POTABLE"}),
        "integrated_pipeline": bool(bundle.get("integrated")),
        "preprocessing_connected": bool(bundle.get("integrated") or bundle.get("preprocessing") is not None),
        "shap_supported": True,
    }


def _transform(frame: pd.DataFrame, bundle: dict) -> Any:
    data: Any = frame
    if bundle.get("integrated"):
        return data
    preprocessing = bundle.get("preprocessing")
    scaler = bundle.get("scaler")
    if preprocessing is not None:
        if hasattr(preprocessing, "transform_features") and callable(preprocessing.transform_features):
            data = preprocessing.transform_features(data)
        elif hasattr(preprocessing, "transform") and callable(preprocessing.transform):
            data = preprocessing.transform(data)
        elif hasattr(preprocessing, "preprocess_frame") and callable(preprocessing.preprocess_frame):
            data = preprocessing.preprocess_frame(data)
        if isinstance(data, np.ndarray) and bundle.get("features"):
            data = pd.DataFrame(data, columns=bundle["features"])
        elif isinstance(data, pd.DataFrame) and bundle.get("features"):
            missing = [f for f in bundle["features"] if f not in data.columns]
            if not missing:
                data = data.loc[:, bundle["features"]]
    if scaler is not None:
        if hasattr(scaler, "transform") and callable(scaler.transform):
            data = scaler.transform(data)
        if isinstance(data, np.ndarray) and bundle.get("features"):
            data = pd.DataFrame(data, columns=bundle["features"])
    return data



def _prediction_error(exc: Exception, bundle: dict, subject: str) -> HTTPException:
    message = str(exc)
    missing_value_failure = any(
        marker in message.lower()
        for marker in ("contains nan", "missing values", "input x contains nan", "does not accept missing")
    )
    if missing_value_failure and not bundle.get("integrated") and bundle.get("preprocessing") is None:
        return HTTPException(
            status_code=422,
            detail={
                "message": f"{subject} contains missing values, but the selected production artifact has no fitted preprocessing object.",
                "model_name": bundle.get("model_name"),
                "artifact_file": bundle.get("config", {}).get("artifact_file"),
                "required_action": "Install the exact preprocessing object fitted on the production model's training data, or explicitly promote a fitted estimator pipeline. AquaGuard will not fit an imputer on prediction data.",
            },
        )
    return HTTPException(
        status_code=422,
        detail=f"{subject} could not pass through the selected production preprocessing/model pipeline: {message}",
    )


def _confidence(model: Any, transformed: Any) -> tuple[float | None, str | None]:
    if not hasattr(model, "predict_proba"):
        return None, None
    try:
        probabilities = np.asarray(model.predict_proba(transformed), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[0] > 0:
            # This is the model's reported maximum class probability, not a calibrated guarantee.
            return float(np.max(probabilities[0])), "predict_proba (model-reported; calibration is not implied)"
    except Exception:
        pass
    return None, None


def predict_frame(frame: pd.DataFrame, *, raise_unavailable: bool = True) -> dict:
    bundle = get_model_bundle()
    if not bundle.get("available"):
        if raise_unavailable:
            raise HTTPException(status_code=503, detail=f"Production prediction is unavailable: {bundle.get('reason')}")
        return {"available": False, "reason": bundle.get("reason")}
    features = bundle["features"]
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Dataset is missing production model features.", "missing_features": missing, "required_features": features})
def _map_label(prediction: Any, config: dict) -> tuple[str, str]:
    labels = config.get("target_labels") or {"0": "NON-POTABLE", "1": "POTABLE"}
    str_val = str(prediction).strip()
    if str_val in labels:
        label = labels[str_val]
    elif str_val in ("1", "1.0", "True", "true", "potable"):
        label = "POTABLE"
    elif str_val in ("0", "0.0", "False", "false", "non-potable", "non_potable"):
        label = "NON-POTABLE"
    else:
        label = str(prediction)

    status_text = f"{label} — predicted {'potable' if label == 'POTABLE' else 'non-potable'} by the ML model"
    return label, status_text


def predict_frame(frame: pd.DataFrame, *, raise_unavailable: bool = True) -> dict:
    bundle = get_model_bundle()
    if not bundle.get("available"):
        if raise_unavailable:
            raise HTTPException(status_code=503, detail=f"Production prediction is unavailable: {bundle.get('reason')}")
        return {"available": False, "reason": bundle.get("reason")}
    features = bundle["features"]
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Dataset is missing production model features.", "missing_features": missing, "required_features": features})
    X = frame.loc[:, features].copy(deep=True)
    transformed = _transform(X, bundle)
    try:
        predictions = bundle["model"].predict(transformed)
    except Exception as exc:
        raise _prediction_error(exc, bundle, "The supplied features") from exc

    config = bundle.get("config", {})
    mapped_labels = []
    status_texts = []
    for pred in predictions:
        lbl, txt = _map_label(pred, config)
        mapped_labels.append(lbl)
        status_texts.append(txt)

    result = {
        "available": True,
        "model_name": bundle["model_name"],
        "predictions": [json_safe(x) for x in predictions],
        "predicted_labels": mapped_labels,
        "status_texts": status_texts,
        "confidence": None,
        "confidence_method": None,
        "potable_count": sum(1 for l in mapped_labels if l == "POTABLE"),
        "non_potable_count": sum(1 for l in mapped_labels if l == "NON-POTABLE"),
        "safety_notice": "A model prediction is not a certified drinking-water safety assessment. Laboratory testing and applicable drinking-water standards are necessary to establish safety.",
    }
    if hasattr(bundle["model"], "predict_proba"):
        try:
            probabilities = np.asarray(bundle["model"].predict_proba(transformed), dtype=float)
            result["confidence"] = [float(np.max(row)) for row in probabilities]
            result["confidence_method"] = "predict_proba (model-reported; calibration is not implied)"
        except Exception:
            pass
    return result


def predict_one(features: dict[str, float]) -> dict:
    from datetime import datetime, timezone
    from backend.services.shap_service import explain_prediction

    bundle = get_model_bundle()
    if not bundle.get("available"):
        raise HTTPException(
            status_code=503,
            detail=f"AI prediction unavailable — trained model not configured: {bundle.get('reason')}"
        )
    expected = bundle["features"]
    missing = [name for name in expected if name not in features]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Input is missing production model features.", "missing_features": missing, "required_features": expected})
    frame = pd.DataFrame([{name: features[name] for name in expected}], columns=expected)
    transformed = _transform(frame, bundle)
    try:
        prediction = bundle["model"].predict(transformed)[0]
    except Exception as exc:
        raise _prediction_error(exc, bundle, "The input") from exc

    config = bundle.get("config", {})
    predicted_label, status_text = _map_label(prediction, config)
    confidence, method = _confidence(bundle["model"], transformed)

    class_probs = None
    if hasattr(bundle["model"], "predict_proba"):
        try:
            raw_probs = bundle["model"].predict_proba(transformed)[0]
            if len(raw_probs) == 2:
                class_probs = {
                    "NON-POTABLE": round(float(raw_probs[0]), 4),
                    "POTABLE": round(float(raw_probs[1]), 4),
                }
            else:
                class_probs = {f"class_{i}": round(float(p), 4) for i, p in enumerate(raw_probs)}
        except Exception:
            pass

    units = config.get("feature_units", {})
    inputs_with_units = [
        {"feature": name, "value": features[name], "unit": units.get(name, "")}
        for name in expected
    ]

    # Compute genuine SHAP explanation
    shap_explanation = explain_prediction(
        model=bundle["model"],
        preprocessor=bundle.get("preprocessing"),
        raw_features=features,
        transformed_frame=transformed,
        features=expected,
        units=units,
        target_labels=config.get("target_labels"),
    )

    return {
        "model_name": bundle["model_name"],
        "model_status": "trained and active",
        "prediction": json_safe(prediction),
        "predicted_label": predicted_label,
        "status_text": status_text,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "confidence_method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_readings": inputs_with_units,
        "shap_explanation": shap_explanation,
        "safety_notice": "A model prediction is not a certified drinking-water safety assessment. Laboratory testing and applicable drinking-water standards are necessary to establish safety.",
        "warning": "Confidence is shown only when the artifact exposes predict_proba; a probability is not necessarily calibrated.",
    }


def invalidate_model_cache() -> None:
    global _CACHE, _CACHE_SIGNATURE
    _CACHE = None
    _CACHE_SIGNATURE = None
