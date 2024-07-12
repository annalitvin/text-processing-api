from pydantic import BaseModel


class TextItem(BaseModel):
    method: str
    line1: str
    line2: str


class SimilarityCreate(BaseModel):
    method: str
    line1: str
    line2: str
    similarity: int