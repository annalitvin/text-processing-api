from pydantic import BaseModel, Field, StrictStr


class PredictRequest(BaseModel):
    input_text: StrictStr = Field(..., title="input_text", description="Input text", examples=["Input text for ML"])


class PredictResponse(BaseModel):
    result: float = Field(..., title="result", description="Predict value", examples=[0.9])
