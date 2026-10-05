from fastapi import APIRouter

from backend.schemas.clustering import ClusteringRequest
from backend.services.clustering_service import compare_clustering, run_hierarchical, run_kmeans
from backend.services.experiment_data_service import load_preprocessed_dataset

router = APIRouter(prefix="/clustering")


def _with_preprocessing(result: dict, report: dict) -> dict:
    result["preprocessing"] = report
    if "kmeans" in result:
        result["kmeans"]["preprocessing"] = report
    if "hierarchical" in result:
        result["hierarchical"]["preprocessing"] = report
    return result


@router.post("/kmeans")
def kmeans(request: ClusteringRequest):
    frame, _, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    return _with_preprocessing(run_kmeans(frame, request.dataset_id, request.features, request.n_clusters), preprocessing)


@router.post("/hierarchical")
def hierarchical(request: ClusteringRequest):
    frame, _, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    return _with_preprocessing(run_hierarchical(frame, request.dataset_id, request.features, request.n_clusters), preprocessing)


@router.post("/compare")
def compare(request: ClusteringRequest):
    frame, _, preprocessing = load_preprocessed_dataset(request.dataset_id, request.preprocessing_sha256)
    return _with_preprocessing(compare_clustering(frame, request.dataset_id, request.features, request.n_clusters), preprocessing)
