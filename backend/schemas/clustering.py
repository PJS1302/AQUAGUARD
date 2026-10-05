from pydantic import BaseModel, Field

class ClusteringRequest(BaseModel):
    dataset_id: str
    preprocessing_sha256: str | None = None
    features: list[str] = Field(..., min_length=1)
    n_clusters: int = Field(default=3, ge=2, le=20)
