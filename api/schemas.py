
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: int
    label_name: str
    explanation: str