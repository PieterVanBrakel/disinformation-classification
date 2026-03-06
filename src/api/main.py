# api/main.py

# Standard library
from fastapi import FastAPI

# Local imports
from api.schemas import ArticleRequest, PredictionResponse
from disinformation_classification.predict import predict

# Create FastAPI app
app = FastAPI(title="Fake News Detection API")


@app.get("/")
def health_check():
    """
    Simple endpoint to verify that the API is running.
    """
    return {"status": "API running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_article(request: ArticleRequest):
    """
    Predict whether an article contains disinformation.
    """

    result = predict(request.text)

    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
    )