"""Explainable AI (XAI) service using SHAP for water quality predictions.

Provides local feature attribution (waterfall / bar contributions) and global
feature importance rankings without fabricating values.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EXPLAINER_CACHE: dict[str, Any] = {}
_BACKGROUND_CACHE: pd.DataFrame | None = None


def _get_background_data(preprocessor: Any, features: list[str]) -> pd.DataFrame | None:
    """Load and preprocess background water quality data for SHAP explainer baselines."""
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is not None:
        return _BACKGROUND_CACHE

    # Search for available background datasets
    candidates = [
        Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo" / "demo_water_quality.csv",
        Path(__file__).resolve().parents[2] / "demo_outputs" / "demo_water_quality_complete_cases.csv",
    ]

    for path in candidates:
        if path.is_file():
            try:
                raw_df = pd.read_csv(path)
                avail_features = [f for f in features if f in raw_df.columns]
                if len(avail_features) == len(features):
                    sample_df = raw_df[features].dropna()
                    if len(sample_df) < 10:
                        sample_df = raw_df[features].fillna(raw_df[features].median())
                    if preprocessor is not None and hasattr(preprocessor, "transform_features"):
                        processed = preprocessor.transform_features(sample_df)
                    elif preprocessor is not None and hasattr(preprocessor, "transform"):
                        processed = preprocessor.transform(sample_df)
                    else:
                        processed = sample_df
                    if isinstance(processed, np.ndarray):
                        processed = pd.DataFrame(processed, columns=features)
                    _BACKGROUND_CACHE = processed.head(100)
                    return _BACKGROUND_CACHE
            except Exception as exc:
                logger.warning("Could not build background data from %s: %s", path, exc)

    return None


def _create_explainer(model: Any, background_df: pd.DataFrame | None) -> Any:
    """Create a task-appropriate SHAP explainer for the given estimator."""
    import shap

    model_type_name = type(model).__name__.lower()

    # Tree-based models: RandomForest, DecisionTree, ExtraTrees, GradientBoosting, XGBoost
    if any(k in model_type_name for k in ("forest", "tree", "xgb", "boost", "lgbm")):
        try:
            return shap.TreeExplainer(model)
        except Exception as exc:
            logger.info("TreeExplainer fallback to generic Explainer: %s", exc)

    # Linear models: LogisticRegression, LinearRegression, Ridge, Lasso
    if any(k in model_type_name for k in ("logistic", "linear", "ridge", "lasso", "sgd")):
        if background_df is not None:
            try:
                return shap.LinearExplainer(model, background_df)
            except Exception as exc:
                logger.info("LinearExplainer fallback to generic Explainer: %s", exc)

    # Generic explainer
    if background_df is not None:
        try:
            # Use predict_proba for classification when available
            predict_fn = getattr(model, "predict_proba", model.predict)
            return shap.Explainer(predict_fn, background_df)
        except Exception as exc:
            logger.info("Explainer creation failed: %s", exc)

    return None


def get_explainer(model: Any, preprocessor: Any, features: list[str]) -> Any:
    """Retrieve or build a cached SHAP explainer for the installed model."""
    model_id = id(model)
    cache_key = f"{model_id}_{len(features)}"

    if cache_key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[cache_key]

    bg_data = _get_background_data(preprocessor, features)
    explainer = _create_explainer(model, bg_data)

    if explainer is not None:
        _EXPLAINER_CACHE[cache_key] = explainer

    return explainer


def _generate_optional_llm_advisory(
    target_class: str,
    predicted_label: str,
    summary_text: str,
    contributions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Generate an optional LLM-powered expert narrative if an API key (e.g. GEMINI_API_KEY or OPENAI_API_KEY) is configured.

    If neither key is configured, returns None so that ML and SHAP work purely locally without network dependencies.
    """
    import os

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not gemini_key and not openai_key:
        return None

    top_pos = [c["feature"] for c in contributions if c.get("direction") == "positive"][:2]
    top_neg = [c["feature"] for c in contributions if c.get("direction") == "negative"][:2]

    prompt = (
        f"AquaGuard Water Quality Assessment:\n"
        f"Prediction: {predicted_label} (Target class: {target_class})\n"
        f"Key positive SHAP factors: {', '.join(top_pos) if top_pos else 'None'}\n"
        f"Key negative SHAP factors: {', '.join(top_neg) if top_neg else 'None'}\n"
        f"Summary: {summary_text}\n"
        f"In 2 brief sentences, provide an expert biochemical interpretation and what filtration/treatment is recommended."
    )

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = getattr(response, "text", "")
            if text:
                return {
                    "provider": "Google Gemini",
                    "advisory": text.strip(),
                }
        except Exception as exc:
            logger.warning("Optional Gemini API call failed: %s", exc)

    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            text = response.choices[0].message.content or ""
            if text:
                return {
                    "provider": "OpenAI",
                    "advisory": text.strip(),
                }
        except Exception as exc:
            logger.warning("Optional OpenAI API call failed: %s", exc)

    return None


def explain_prediction(
    model: Any,
    preprocessor: Any,
    raw_features: dict[str, float],
    transformed_frame: pd.DataFrame,
    features: list[str],
    units: dict[str, str] | None = None,
    target_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute local SHAP feature attributions for a single prediction instance.

    Returns true calculated SHAP values without fabrication.
    """
    if units is None:
        units = {}
    if target_labels is None:
        target_labels = {"0": "NON-POTABLE", "1": "POTABLE"}

    try:
        import shap  # noqa: F401
    except ImportError:
        return {
            "supported": False,
            "reason": "SHAP library is not installed in the Python environment.",
            "contributions": [],
        }

    try:
        explainer = get_explainer(model, preprocessor, features)
        if explainer is None:
            return {
                "supported": False,
                "reason": f"No compatible SHAP explainer could be constructed for model type {type(model).__name__}.",
                "contributions": [],
            }

        # Calculate SHAP explanation for the single instance
        explanation = explainer(transformed_frame)

        # Extract values
        shap_vals = explanation.values
        base_vals = explanation.base_values

        # Handle different dimensionalities
        # 1. Shape (1, n_features, n_classes) -> take class 1 (POTABLE)
        if hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
            # class 1 corresponds to POTABLE in standard potability datasets
            class_idx = 1 if shap_vals.shape[2] > 1 else 0
            instance_shap = shap_vals[0, :, class_idx]
            base_val = float(base_vals[0, class_idx] if base_vals.ndim > 1 else base_vals[0])
        # 2. Shape (1, n_features) -> binary decision margin / log-odds
        elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
            instance_shap = shap_vals[0, :]
            base_val = float(base_vals[0] if hasattr(base_vals, "__len__") else base_vals)
        else:
            instance_shap = np.asarray(shap_vals).flatten()
            base_val = float(base_vals) if np.isscalar(base_vals) else float(base_vals[0])

        contributions = []
        positive_influences = []
        negative_influences = []

        for idx, feature_name in enumerate(features):
            raw_val = raw_features.get(feature_name)
            val = float(raw_val) if raw_val is not None else 0.0
            sv = float(instance_shap[idx]) if idx < len(instance_shap) else 0.0
            unit = units.get(feature_name, "")

            # Direction toward POTABLE (positive SHAP) vs toward NON-POTABLE (negative SHAP)
            if sv > 0:
                direction = "increases_potability"
                influence_label = "POTABLE"
                positive_influences.append((feature_name, sv))
            else:
                direction = "decreases_potability"
                influence_label = "NON-POTABLE"
                negative_influences.append((feature_name, sv))

            contributions.append({
                "feature": feature_name,
                "value": round(val, 3),
                "unit": unit,
                "shap_value": round(sv, 4),
                "abs_shap": round(abs(sv), 4),
                "direction": direction,
                "influence": influence_label,
            })

        # Sort contributions by absolute magnitude (highest influence first)
        contributions.sort(key=lambda item: item["abs_shap"], reverse=True)

        # Calculate max absolute value to compute relative impact percentages
        max_abs = max((c["abs_shap"] for c in contributions), default=1.0) or 1.0
        for c in contributions:
            c["relative_impact_percent"] = round((c["abs_shap"] / max_abs) * 100, 1)

        # Compute sequential waterfall data
        running_sum = base_val
        waterfall_steps = [{"label": "Baseline Expected Value", "value": round(base_val, 4), "cumulative": round(running_sum, 4), "type": "base"}]
        for c in contributions:
            running_sum += c["shap_value"]
            waterfall_steps.append({
                "label": c["feature"],
                "value": c["shap_value"],
                "cumulative": round(running_sum, 4),
                "direction": c["direction"],
                "type": "feature",
            })
        waterfall_steps.append({
            "label": "Model Decision Score",
            "value": round(running_sum, 4),
            "cumulative": round(running_sum, 4),
            "type": "final",
        })

        # Generate human-readable narrative summary
        positive_influences.sort(key=lambda x: abs(x[1]), reverse=True)
        negative_influences.sort(key=lambda x: abs(x[1]), reverse=True)

        summary_parts = []
        if positive_influences:
            top_pos = [f"{name} (+{round(val, 3)})" for name, val in positive_influences[:2]]
            summary_parts.append(f"{', '.join(top_pos)} pushed prediction toward POTABLE")
        if negative_influences:
            top_neg = [f"{name} ({round(val, 3)})" for name, val in negative_influences[:2]]
            summary_parts.append(f"{', '.join(top_neg)} pulled prediction toward NON-POTABLE")

        summary_text = "; ".join(summary_parts) if summary_parts else "Features contributed evenly across baseline."

        # Optional Generative AI explanation if user configured an API key
        ai_advisory = _generate_optional_llm_advisory(target_class="POTABLE", predicted_label=target_labels.get("1" if running_sum >= 0 else "0", "POTABLE"), summary_text=summary_text, contributions=contributions)

        return {
            "supported": True,
            "base_value": round(base_val, 4),
            "decision_score": round(running_sum, 4),
            "target_class": "POTABLE",
            "target_labels": target_labels,
            "contributions": contributions,
            "waterfall": waterfall_steps,
            "top_positive": [item[0] for item in positive_influences[:3]],
            "top_negative": [item[0] for item in negative_influences[:3]],
            "summary_text": summary_text,
            "ai_expert_advisory": ai_advisory,
            "disclaimer": "SHAP explains how the mathematical model reached its decision; it does not independently establish certified drinking water safety.",
        }

    except Exception as exc:
        logger.warning("SHAP calculation error: %s", exc)
        return {
            "supported": False,
            "reason": f"SHAP explanation calculation failed: {exc}",
            "contributions": [],
        }


def get_global_feature_importance(
    model: Any,
    preprocessor: Any,
    features: list[str],
    units: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute global feature importance ranking using mean absolute SHAP values."""
    if units is None:
        units = {}

    bg_data = _get_background_data(preprocessor, features)
    if bg_data is None:
        return {"available": False, "reason": "No background water quality dataset available for global SHAP baseline.", "items": []}

    try:
        explainer = get_explainer(model, preprocessor, features)
        if explainer is None:
            return {"available": False, "reason": "Could not create SHAP explainer.", "items": []}

        # Calculate SHAP values for background sample
        explanation = explainer(bg_data.head(50))
        shap_vals = explanation.values

        if hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
            class_idx = 1 if shap_vals.shape[2] > 1 else 0
            mean_abs = np.mean(np.abs(shap_vals[:, :, class_idx]), axis=0)
        elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
        else:
            mean_abs = np.mean(np.abs(shap_vals), axis=0)

        items = []
        for idx, feat in enumerate(features):
            score = float(mean_abs[idx]) if idx < len(mean_abs) else 0.0
            items.append({
                "feature": feat,
                "unit": units.get(feat, ""),
                "mean_abs_shap": round(score, 4),
            })

        items.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        max_score = max((x["mean_abs_shap"] for x in items), default=1.0) or 1.0
        for idx, item in enumerate(items):
            item["rank"] = idx + 1
            item["relative_importance_percent"] = round((item["mean_abs_shap"] / max_score) * 100, 1)

        return {
            "available": True,
            "sample_size": min(len(bg_data), 50),
            "items": items,
            "method": "Mean absolute SHAP value across reference water samples",
        }
    except Exception as exc:
        return {"available": False, "reason": f"Global SHAP analysis failed: {exc}", "items": []}
