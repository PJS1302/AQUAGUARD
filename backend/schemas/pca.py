from pydantic import BaseModel, Field

class PCARequest(BaseModel):
    dataset_id: str
    preprocessing_sha256: str | None = None
    features: list[str] = Field(..., min_length=1)
    n_components: int = Field(default=3, ge=1, le=100)
