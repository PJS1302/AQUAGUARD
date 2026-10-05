from fastapi import APIRouter

from backend.schemas.classification import ClassificationRequest, ProductionSelection
from backend.services.classification_service import compare_classifiers, promote_experiment, run_classification
from backend.services.experiment_data_service import load_preprocessed_dataset

router = APIRouter(prefix="/classification")


def _run(request: ClassificationRequest, algorithm: str):
    frame, metadata, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    result = run_classification(
        frame,
        dataset_id=request.dataset_id,
        dataset_metadata=metadata,
        target=request.target,
        features=request.features,
        algorithm=algorithm,
        folds=request.folds,
        k=request.k,
        n_estimators=request.n_estimators,
        learning_rate=request.learning_rate,
        preprocessing_report=preprocessing,
    )
    result["preprocessing"] = preprocessing
    return result


@router.post("/decision-tree")
def decision_tree(request: ClassificationRequest):
    return _run(request, "decision-tree")


@router.post("/c45")
def c45(request: ClassificationRequest):
    return _run(request, "c45")


@router.post("/naive-bayes")
def naive_bayes(request: ClassificationRequest):
    return _run(request, "naive-bayes")


@router.post("/knn")
def knn(request: ClassificationRequest):
    return _run(request, "knn")


@router.post("/bagging")
def bagging(request: ClassificationRequest):
    return _run(request, "bagging")


@router.post("/adaboost")
def adaboost(request: ClassificationRequest):
    return _run(request, "adaboost")


@router.post("/random-forest")
def random_forest(request: ClassificationRequest):
    return _run(request, "random-forest")


@router.post("/compare")
def compare(request: ClassificationRequest):
    frame, metadata, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    result = compare_classifiers(
        frame,
        dataset_id=request.dataset_id,
        dataset_metadata=metadata,
        target=request.target,
        features=request.features,
        folds=request.folds,
        k=request.k,
        n_estimators=request.n_estimators,
        learning_rate=request.learning_rate,
        preprocessing_report=preprocessing,
    )
    result["preprocessing"] = preprocessing
    return result


@router.post("/production")
def select_production(request: ProductionSelection):
    return promote_experiment(request.experiment_id, request.model_name)
