from __future__ import annotations
from datetime import datetime, timezone
import json
import shutil
import math
import re
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from backend.utils.experiment_data import require_complete_processed_rows, require_finite_numeric

from backend.config import EXPERIMENT_DIR
from backend.services.database_service import record_model_run
from backend.utils.dataset_validator import json_safe


def _entropy(y: np.ndarray) -> float:
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-15)))


class C45Classifier(ClassifierMixin, BaseEstimator):
    """A numeric-feature C4.5-style tree using entropy and information gain ratio.

    At each node, thresholds are evaluated using entropy. Candidate splits are
    filtered against mean information gain before the best gain ratio is chosen,
    following C4.5's gain-ratio bias correction rather than CART's Gini criterion.
    """
    def __init__(self, max_depth: int = 8, min_samples_split: int = 4, min_samples_leaf: int = 2, max_thresholds: int = 32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
            values = X.to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
            self.feature_names_in_ = np.asarray([f"x{i}" for i in range(values.shape[1])], dtype=object)
        labels = np.asarray(y)
        if values.ndim != 2 or len(values) != len(labels):
            raise ValueError("X must be a 2D numeric matrix aligned with y")
        if not np.isfinite(values).all():
            raise ValueError("C4.5-style numeric tree requires complete finite feature values")
        self.classes_ = np.unique(labels)
        self.n_features_in_ = values.shape[1]
        self._root = self._build(values, labels, depth=0)
        self.n_features_ = self.n_features_in_
        return self

    def _candidate_thresholds(self, column: np.ndarray) -> np.ndarray:
        unique = np.unique(column)
        if len(unique) <= 1: return np.asarray([])
        mids = (unique[:-1] + unique[1:]) / 2.0
        if len(mids) > self.max_thresholds:
            quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
            mids = np.unique(np.quantile(mids, quantiles))
        return mids

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        parent_entropy = _entropy(y)
        candidates = []
        for feature_index in range(X.shape[1]):
            column = X[:, feature_index]
            for threshold in self._candidate_thresholds(column):
                left_mask = column <= threshold
                n_left = int(left_mask.sum())
                n_right = len(y) - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                p_left = n_left / len(y)
                conditional = p_left * _entropy(y[left_mask]) + (1 - p_left) * _entropy(y[~left_mask])
                gain = parent_entropy - conditional
                split_info = -(p_left * math.log2(p_left) + (1 - p_left) * math.log2(1 - p_left))
                if split_info > 0 and gain > 1e-12:
                    candidates.append((gain, gain / split_info, feature_index, float(threshold)))
        if not candidates:
            return None
        mean_gain = float(np.mean([c[0] for c in candidates]))
        eligible = [c for c in candidates if c[0] >= mean_gain - 1e-12]
        return max(eligible or candidates, key=lambda c: c[1])

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        values, counts = np.unique(y, return_counts=True)
        majority = values[int(np.argmax(counts))]
        node = {"label": majority, "samples": int(len(y)), "feature": None, "threshold": None, "left": None, "right": None}
        if len(values) <= 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        _, gain_ratio, feature_index, threshold = split
        mask = X[:, feature_index] <= threshold
        if mask.sum() == 0 or (~mask).sum() == 0:
            return node
        node.update({"feature": int(feature_index), "threshold": float(threshold), "gain_ratio": float(gain_ratio)})
        node["left"] = self._build(X[mask], y[mask], depth + 1)
        node["right"] = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, row):
        node = self._root
        while node["feature"] is not None:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["label"]

    def predict(self, X):
        values = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match the fitted C4.5-style tree")
        return np.asarray([self._predict_one(row) for row in values])

    def tree_summary(self, node=None, depth=0, limit=4):
        node = self._root if node is None else node
        if node["feature"] is None or depth >= limit:
            return {"leaf": json_safe(node["label"]), "samples": node["samples"]}
        feature_name = str(self.feature_names_in_[node["feature"]])
        return {
            "feature": feature_name, "threshold": node["threshold"], "gain_ratio": node.get("gain_ratio"),
            "samples": node["samples"], "left_le": self.tree_summary(node["left"], depth + 1, limit),
            "right_gt": self.tree_summary(node["right"], depth + 1, limit),
        }


def _make_estimator(name: str, numeric: list[str], categorical: list[str], *, k: int = 5, n_estimators: int = 100, learning_rate: float = 0.8):
    if categorical:
        raise HTTPException(
            status_code=422,
            detail=(
                "The selected processed features are not all numeric. Apply the user's "
                "actual encoding in the installed preprocessing pipeline; AquaGuard will not encode or scale them."
            ),
        )
    if not numeric:
        raise HTTPException(status_code=422, detail="Select at least one numeric feature from the user's processed dataset.")
    if name == "c45":
        return C45Classifier()
    if name == "decision-tree":
        return DecisionTreeClassifier(random_state=42, class_weight=None)
    if name == "naive-bayes":
        return GaussianNB()
    if name == "knn":
        # KNN consumes exactly the processed values; AquaGuard adds no scaler.
        return KNeighborsClassifier(n_neighbors=int(k), weights="distance")
    if name == "bagging":
        return BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=int(n_estimators), random_state=42, n_jobs=-1)
    if name == "adaboost":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=int(n_estimators), learning_rate=float(learning_rate), random_state=42)
    if name == "random-forest":
        return RandomForestClassifier(n_estimators=int(n_estimators), random_state=42, n_jobs=-1, class_weight="balanced")
    raise HTTPException(status_code=400, detail=f"Unknown classification algorithm '{name}'.")


def _prepare(frame: pd.DataFrame, target: str, features: list[str]):
    if target not in frame.columns:
        raise HTTPException(status_code=422, detail=f"Target column '{target}' was not found in the user's processed dataset.")
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Selected processed classification feature(s) are missing.", "missing_features": missing})
    if not features or len(set(features)) != len(features):
        raise HTTPException(status_code=422, detail="Select one or more distinct processed input features.")
    if target in features:
        raise HTTPException(status_code=422, detail="The target cannot also be an input feature.")

    columns = features + [target]
    data = require_complete_processed_rows(frame, columns, task="Classification")
    if len(data) > 100_000:
        raise HTTPException(status_code=413, detail="Supervised ML-lab experiments are limited to 100,000 complete processed rows per run.")
    y = data[target].copy()
    if y.nunique() < 2:
        raise HTTPException(status_code=422, detail="Classification requires at least two target classes.")
    if pd.api.types.is_numeric_dtype(y) and int(y.nunique()) > 20 and int(y.nunique()) / max(1, len(y)) > 0.2:
        raise HTTPException(status_code=422, detail="The selected numerical target has many distinct values and may be continuous. Choose Classification only for discrete labels; use Regression for continuous targets.")
    numeric = data[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    if categorical:
        raise HTTPException(
            status_code=422,
            detail=(
                "The user's preprocessing output still contains categorical input feature(s): "
                + ", ".join(categorical)
                + ". AquaGuard will not automatically encode them."
            ),
        )
    finite_columns = numeric + ([target] if pd.api.types.is_numeric_dtype(y) else [])
    require_finite_numeric(data, finite_columns, task="Classification")
    if len(data) * max(1, len(numeric)) > 20_000_000:
        raise HTTPException(status_code=413, detail="Selected processed features exceed the supervised lab memory guard.")
    return data[features].copy(deep=True), y.copy(deep=True), numeric, categorical


def _split_indices(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = y.value_counts()
    classes = len(counts)
    if int(counts.min()) < 2:
        raise HTTPException(status_code=422, detail="Every target class needs at least two complete samples for a stratified holdout evaluation.")
    n_test = max(int(math.ceil(len(y) * 0.2)), classes)
    if len(y) - n_test < classes:
        raise HTTPException(status_code=422, detail="Not enough samples per class for a stratified train/test split.")
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=n_test, random_state=42, stratify=y)
    return train, test


def _score(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    labels = list(pd.unique(pd.concat([pd.Series(y_test), pd.Series(predictions)])))
    labels = sorted(labels, key=lambda value: str(value))
    cm = confusion_matrix(y_test, predictions, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": {"labels": [json_safe(v) for v in labels], "values": cm.tolist()},
        "predictions": [json_safe(v) for v in predictions[:200]],
        "actual": [json_safe(v) for v in np.asarray(y_test)[:200]],
    }


def _cv_score(estimator, X: pd.DataFrame, y: pd.Series, requested: int) -> dict:
    min_class = int(y.value_counts().min())
    folds = min(int(requested), min_class)
    if folds < 2:
        return {"folds": 0, "fold_scores": [], "mean_score": None, "std_dev": None, "metric": "accuracy", "message": "K-Fold is not valid: a class has fewer than two complete samples."}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(estimator, X, y, cv=splitter, scoring="accuracy", error_score="raise", n_jobs=None)
        return {"folds": folds, "fold_scores": [float(s) for s in scores], "mean_score": float(np.mean(scores)), "std_dev": float(np.std(scores, ddof=0)), "metric": "accuracy"}
    except Exception as exc:
        return {"folds": folds, "fold_scores": [], "mean_score": None, "std_dev": None, "metric": "accuracy", "message": f"K-Fold could not be calculated: {exc}"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _save_experiment(experiment_id: str, name: str, estimator, features: list[str], target: str, dataset_id: str, metadata: dict, preprocessing_report: dict | None = None) -> str:
    folder = EXPERIMENT_DIR / experiment_id
    folder.mkdir(parents=True, exist_ok=True)
    artifact_file = f"{_slug(name)}.joblib"
    joblib.dump(estimator, folder / artifact_file)
    meta_path = folder / "metadata.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        data = {"experiment_id": experiment_id, "dataset_id": dataset_id, "target": target, "features": features, "is_demo": bool(metadata.get("is_demo")), "created_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    if preprocessing_report:
        data["preprocessing"] = {
            "pipeline_name": preprocessing_report.get("pipeline_name"),
            "artifact_file": preprocessing_report.get("artifact_file"),
            "artifact_sha256": preprocessing_report.get("artifact_sha256"),
            "pipeline_steps": preprocessing_report.get("pipeline_steps", []),
        }
    data["models"][name] = {"artifact": artifact_file, "features": features, "target": target}
    meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(folder / artifact_file)


def run_classification(frame: pd.DataFrame, *, dataset_id: str, dataset_metadata: dict, target: str, features: list[str] | None, algorithm: str, folds: int = 5, k: int = 5, n_estimators: int = 100, learning_rate: float = 0.8, preprocessing_report: dict | None = None) -> dict:
    selected = [str(c) for c in frame.columns if c != target] if features is None else features
    X, y, numeric, categorical = _prepare(frame, target, selected)
    if algorithm == "knn" and int(k) > len(X) * 0.8:
        raise HTTPException(status_code=422, detail=f"K must be no greater than the training sample count (currently at most {max(1, int(len(X) * .8))}).")
    if n_estimators < 1 or n_estimators > 1000:
        raise HTTPException(status_code=422, detail="Number of estimators must be between 1 and 1000.")
    if not (0 < learning_rate <= 10):
        raise HTTPException(status_code=422, detail="AdaBoost learning rate must be between 0 and 10.")
    train_idx, test_idx = _split_indices(y)
    estimator = _make_estimator(algorithm, numeric, categorical, k=k, n_estimators=n_estimators, learning_rate=learning_rate)
    try:
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = _score(estimator, X.iloc[test_idx], y.iloc[test_idx])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{algorithm} could not be fitted: {exc}") from exc
    cv = _cv_score(clone(_make_estimator(algorithm, numeric, categorical, k=k, n_estimators=n_estimators, learning_rate=learning_rate)), X, y, folds)
    # A separate refit on all complete rows is retained for optional, explicit promotion.
    full_model = _make_estimator(algorithm, numeric, categorical, k=k, n_estimators=n_estimators, learning_rate=learning_rate)
    try:
        full_model.fit(X, y)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"The evaluated model could not be refitted on all complete rows: {exc}") from exc
    experiment_id = str(uuid4())
    _save_experiment(experiment_id, algorithm, full_model, selected, target, dataset_id, dataset_metadata, preprocessing_report)
    from backend.services.user_preprocessing_service import preprocessing_status
    prep_status = preprocessing_status()
    promotion_ready = (
        not bool(dataset_metadata.get("is_demo"))
        and bool(prep_status.get("supports_production_inference"))
        and isinstance(prep_status.get("input_features"), list)
        and all(isinstance(item, str) for item in prep_status.get("input_features", []))
    )
    ineligible_reason = None
    if dataset_metadata.get("is_demo"):
        ineligible_reason = "Synthetic demo data is never eligible for production promotion."
    elif not promotion_ready:
        ineligible_reason = "The installed user preprocessing artifact must also expose transform_features(frame) and declare exact raw input_features before this lab model can be promoted."
    result = {
        "algorithm": algorithm, "display_name": {"decision-tree": "Decision Tree / CART", "c45": "C4.5-style (gain ratio)", "naive-bayes": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbors", "bagging": "Bagging", "adaboost": "AdaBoost", "random-forest": "Random Forest"}.get(algorithm, algorithm),
        "dataset_id": dataset_id, "target": target, "features": selected,
        "numeric_features": numeric, "categorical_features": categorical,
        "complete_rows": int(len(X)),
        "metrics": {k: v for k, v in scores.items() if k not in {"predictions", "actual"}},
        "actual": scores["actual"], "predicted": scores["predictions"],
        "cross_validation": cv, "experiment_id": experiment_id,
        "production_eligible": promotion_ready,
        "production_ineligible_reason": ineligible_reason,
    }
    if algorithm == "c45":
        result["tree"] = estimator.tree_summary()
        result["method_note"] = "Numeric thresholds are ranked by entropy-based information gain ratio; candidate splits are filtered by mean information gain. This is not CART/Gini."
    if algorithm == "decision-tree":
        result["tree"] = export_text(full_model, feature_names=selected, max_depth=4, show_weights=True)
        result["tree_note"] = "CART tree summary (Gini criterion); truncated at depth 4 for display."
    if algorithm == "random-forest":
        classifier = full_model
        names = selected
        result["feature_importance"] = sorted(
            [{"feature": str(name), "importance": float(value)} for name, value in zip(names, classifier.feature_importances_)],
            key=lambda item: item["importance"], reverse=True,
        )[:30]
    record_model_run(result["display_name"], dataset_id, result["metrics"] | {"cross_validation": cv})
    return result


def compare_classifiers(frame: pd.DataFrame, *, dataset_id: str, dataset_metadata: dict, target: str, features: list[str] | None, folds: int, k: int, n_estimators: int, learning_rate: float, preprocessing_report: dict | None = None) -> dict:
    algorithms = ["decision-tree", "c45", "naive-bayes", "knn", "bagging", "adaboost", "random-forest"]
    results, errors = [], []
    group_id = str(uuid4())
    for algorithm in algorithms:
        try:
            item = run_classification(frame, dataset_id=dataset_id, dataset_metadata=dataset_metadata, target=target, features=features, algorithm=algorithm, folds=folds, k=k, n_estimators=n_estimators, learning_rate=learning_rate, preprocessing_report=preprocessing_report)
            # Keep one comparison id across its algorithms, while each run artifact remains addressable by its own id.
            item["comparison_id"] = group_id
            results.append(item)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            errors.append({"algorithm": algorithm, "message": detail})
    if not results:
        raise HTTPException(status_code=422, detail={"message": "No classification model completed.", "errors": errors})
    return {"comparison_id": group_id, "results": results, "errors": errors, "production_eligible": all(bool(x.get("production_eligible")) for x in results)}


def promote_experiment(experiment_id: str, model_name: str) -> dict:
    from backend.config import MODELS_DIR
    from backend.services.user_preprocessing_service import UserPipelinePredictor, get_user_pipeline
    from backend.utils.model_loader import invalidate_model_cache

    folder = (EXPERIMENT_DIR / experiment_id).resolve()
    if EXPERIMENT_DIR.resolve() not in folder.parents or not folder.exists():
        raise HTTPException(status_code=404, detail="Experiment artifacts were not found.")
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Experiment metadata was not found.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("is_demo"):
        raise HTTPException(status_code=409, detail="A model trained from AquaGuard's Demo Dataset cannot be promoted as a production model. Upload validated project data and re-evaluate it first.")
    model_info = metadata.get("models", {}).get(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail="That model is not part of the selected experiment.")
    artifact_path = (folder / model_info["artifact"]).resolve()
    if folder not in artifact_path.parents or not artifact_path.exists():
        raise HTTPException(status_code=404, detail="The selected model artifact is missing.")

    pipeline, pipeline_metadata, pipeline_status = get_user_pipeline()
    saved_preprocessing = metadata.get("preprocessing") or {}
    if saved_preprocessing.get("artifact_sha256") != pipeline_status.get("artifact_sha256"):
        raise HTTPException(
            status_code=409,
            detail="The active user preprocessing artifact differs from the one used for this experiment. Re-run the experiment with the current pipeline before promotion.",
        )
    raw_features = pipeline_status.get("input_features")
    if not callable(getattr(pipeline, "transform_features", None)) or not isinstance(raw_features, list) or not raw_features or not all(isinstance(item, str) for item in raw_features):
        raise HTTPException(
            status_code=409,
            detail=(
                "This experiment cannot be promoted safely. The exact user pipeline must "
                "implement transform_features(frame) for raw inference and declare its "
                "ordered raw input_features. No preprocessing was guessed."
            ),
        )

    try:
        estimator = joblib.load(artifact_path)
        production_pipeline = UserPipelinePredictor(pipeline, estimator, model_info["features"])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not assemble the candidate with the installed user pipeline: {exc}") from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_name = f"production_model_promoted_{experiment_id[:8]}.pkl"
    target_path = MODELS_DIR / artifact_name
    if target_path.exists():
        raise HTTPException(status_code=409, detail="A promoted artifact with this id already exists; no existing artifact was overwritten.")

    config = {
        "model_name": model_name,
        "features": raw_features,
        "transformed_features": model_info["features"],
        "target_name": model_info["target"],
        "preprocessing_in_pipeline": True,
        "selected_experiment_id": experiment_id,
        "user_preprocessing_artifact": pipeline_status["artifact_file"],
        "selection_note": "Fit on all complete processed rows after a separate holdout and cross-validation evaluation. The installed user pipeline is retained for raw inference.",
    }
    bundle = {"pipeline": production_pipeline, "feature_config": config}
    temporary_artifact = MODELS_DIR / f"{artifact_name}.tmp"
    try:
        joblib.dump(bundle, temporary_artifact)
        temporary_artifact.replace(target_path)
    except Exception as exc:
        temporary_artifact.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save the promoted model bundle: {exc}") from exc

    active_config = MODELS_DIR / "feature_config.json"
    if active_config.exists():
        backup_config = MODELS_DIR / f"feature_config_backup_{stamp}.json"
        shutil.copy2(active_config, backup_config)
    config["artifact_file"] = artifact_name
    temp_config = MODELS_DIR / "feature_config.json.tmp"
    temp_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temp_config.replace(active_config)
    invalidate_model_cache()
    return {
        "status": "promoted",
        "model_name": model_name,
        "features": raw_features,
        "transformed_features": model_info["features"],
        "artifact_file": artifact_name,
        "previous_model_artifacts_preserved": True,
        "warning": "This explicit promotion retains the installed preprocessing pipeline. It does not certify drinking-water safety; validate against the intended deployment population.",
    }
