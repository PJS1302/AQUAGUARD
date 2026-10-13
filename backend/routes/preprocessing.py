from fastapi import APIRouter

from backend.services.user_preprocessing_service import preprocessing_status

router = APIRouter(prefix="/preprocessing")


@router.get("/status")
def get_preprocessing_status(dataset_id: str | None = None):
    return preprocessing_status(dataset_id=dataset_id)

