from fastapi import APIRouter
from backend.database.connection import database_status
from backend.services.ml_service import model_status
from backend.services.user_preprocessing_service import preprocessing_status

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "database": database_status(), "production_model": model_status(), "preprocessing_pipeline": preprocessing_status()}
