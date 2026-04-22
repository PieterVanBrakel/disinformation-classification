from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import yaml
from pathlib import Path

from .schemas import PredictRequest, PredictResponse
from src.data_inference.explain import RAGExplainer
from src.data_inference.report import build_report

from dotenv import load_dotenv
load_dotenv()


LABEL_MAP = {0: "real", 1: "disinformation"}
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    config_path = PROJECT_ROOT / "src" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vec_path = PROJECT_ROOT / config["output"]["vectorizer_path"]
    model_path = PROJECT_ROOT / config["output"]["model_path"]

    if not vec_path.exists() or not model_path.exists():
        raise RuntimeError("Run training pipeline first.")

    model_state["vectorizer"] = joblib.load(vec_path)
    model_state["model"] = joblib.load(model_path)

    # 🧠 RAG engine (NEW)
    model_state["rag"] = RAGExplainer()

    print("Model + RAG loaded.")
    yield
    model_state.clear()


app = FastAPI(title="Disinformation Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):

    vec = model_state["vectorizer"]
    model = model_state["model"]
    rag = model_state["rag"]

    X = vec.transform([request.text])
    label = int(model.predict(X)[0])

    explanation = rag.explain(request.text, str(label))

    report = build_report(
        text=request.text,
        label=label,
        explanation=explanation
    )

    return {
        "text": request.text,
        "label": label,
        "label_name": LABEL_MAP.get(label),
        "explanation": explanation,
        "report": report
    }