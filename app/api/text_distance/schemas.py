from typing import Union

from pydantic import BaseModel


class TextItem(BaseModel):
    method: str
    line1: str
    line2: str


class SimilarityCreate(BaseModel):
    method: str
    line1: str
    line2: str
    similarity: Union[int, float]

    class Config:
        smart_union = True
