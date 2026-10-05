from fastapi import APIRouter

from backend.schemas.pca import PCARequest
from backend.services.experiment_data_service import load_preprocessed_dataset
from backend.services.pca_service import run_pca

router = APIRouter(prefix="/pca")


@router.post("/analyze")
def analyze(request: PCARequest):
    frame, _, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    result = run_pca(frame, request.features, request.n_components, request.dataset_id)
    result["preprocessing"] = preprocessing
    return result
