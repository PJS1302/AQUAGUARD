from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class DeviceCreate(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=128)
    device_name: str = Field(..., min_length=1, max_length=200)
    location: str | None = Field(default=None, max_length=300)

    @field_validator("device_id", "device_name")
    @classmethod
    def non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value must contain a non-whitespace character.")
        return normalized

    @field_validator("location")
    @classmethod
    def normalize_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

class ReadingCreate(BaseModel):
    timestamp: datetime | None = None
    features: dict[str, float] = Field(..., min_length=1)
    source: str = Field(default="sensor", pattern="^(sensor|simulation)$")
