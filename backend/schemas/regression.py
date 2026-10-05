from pydantic import BaseModel, Field

class RegressionRequest(BaseModel):
    dataset_id: str
    preprocessing_sha256: str | None = None
    target: str
    features: list[str] = Field(..., min_length=1)
    degree: int = Field(default=2, ge=2, le=5)
    folds: int = Field(default=5, ge=2, le=10)
    bandwidth: float = Field(default=0.8, gt=0, le=20)
    query: dict[str, float] | None = None
