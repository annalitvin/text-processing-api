from typing import Any

from fastapi import Request
from fastapi import APIRouter

from app.models.predict import PredictRequest, PredictResponse
from .constants import ALGORITHM_NOT_EXISTS_ERROR_MSG
from .text_distance.schemas import TextItem, SimilarityCreate
from .text_distance.similarity_algorithm_factory import TextSimilarityAlgorithmFactory

from fastapi import HTTPException
from fastapi import status

api_router = APIRouter()

@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_text = payload.input_text
    model = request.app.state.model

    predict_value = model.predict(input_text)
    return PredictResponse(result=predict_value)


@api_router.post("/calculate_similarity", name="calculate_similarity", response_model=SimilarityCreate)
async def calculate_similarity(text_item: TextItem):
    method = text_item.method
    try:
        algorithm = TextSimilarityAlgorithmFactory.make(method)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ALGORITHM_NOT_EXISTS_ERROR_MSG)

    line1 = text_item.line1
    line2 = text_item.line2
    similarity = algorithm.similarity(line1, line2)

    return SimilarityCreate(
        method=text_item.method, line1=text_item.line1, line2=text_item.line2, similarity=similarity
    )
