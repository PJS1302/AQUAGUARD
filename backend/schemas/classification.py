from pydantic import BaseModel, Field

class ClassificationRequest(BaseModel):
    dataset_id: str
    preprocessing_sha256: str | None = None
    target: str
    features: list[str] | None = None
    folds: int = Field(default=5, ge=2, le=10)
    k: int = Field(default=5, ge=1, le=200)
    n_estimators: int = Field(default=100, ge=1, le=1000)
    learning_rate: float = Field(default=0.8, gt=0, le=10)

class ProductionSelection(BaseModel):
    experiment_id: str
    model_name: str
