from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., min_length=1)
    source: str = "manual"

class PredictionResult(BaseModel):
    model_name: str
    prediction: str | int | float | bool
    confidence: float | None = None
    confidence_method: str | None = None
    warning: str | None = None
