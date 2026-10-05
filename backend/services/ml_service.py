"""Production prediction helpers; artifact loading lives in backend.utils.model_loader."""
from backend.utils.model_loader import model_status, predict_frame, predict_one

__all__ = ["model_status", "predict_frame", "predict_one"]
