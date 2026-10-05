from datetime import datetime
from pydantic import BaseModel, Field

class DeviceCreate(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=128)
    device_name: str = Field(..., min_length=1, max_length=200)
    location: str | None = Field(default=None, max_length=300)

class ReadingCreate(BaseModel):
    timestamp: datetime | None = None
    features: dict[str, float] = Field(..., min_length=1)
    source: str = Field(default="sensor", pattern="^(sensor|simulation)$")
