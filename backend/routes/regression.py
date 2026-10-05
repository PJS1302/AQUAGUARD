from fastapi import APIRouter

from backend.schemas.regression import RegressionRequest
from backend.services.experiment_data_service import load_preprocessed_dataset
from backend.services.regression_service import run_regression

router = APIRouter(prefix="/regression")


def _run(request: RegressionRequest, algorithm: str):
    frame, _, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    result = run_regression(
        frame,
        dataset_id=request.dataset_id,
        algorithm=algorithm,
        target=request.target,
        features=request.features,
        degree=request.degree,
        folds=request.folds,
        bandwidth=request.bandwidth,
        query=request.query,
    )
    result["preprocessing"] = preprocessing
    return result


@router.post("/linear")
def linear(request: RegressionRequest):
    return _run(request, "linear")


@router.post("/multiple")
def multiple(request: RegressionRequest):
    return _run(request, "multiple")


@router.post("/polynomial")
def polynomial(request: RegressionRequest):
    return _run(request, "polynomial")


@router.post("/lwlr")
def lwlr(request: RegressionRequest):
    return _run(request, "lwlr")
