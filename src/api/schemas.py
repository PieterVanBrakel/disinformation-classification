from pydantic import BaseModel

class ArticleRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: float