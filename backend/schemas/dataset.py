from pydantic import BaseModel


class DatasetUploadInfo(BaseModel):
    id: str
    file_name: str
    row_count: int
    column_count: int
    compatibility_status: str
    is_demo: bool = False
