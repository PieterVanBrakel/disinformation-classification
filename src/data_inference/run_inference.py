import joblib
import yaml
from pathlib import Path

from dotenv import load_dotenv

from .predict import predict
from .explain import RAGExplainer

# Load environment variables (.env)
load_dotenv()


def run_inference():

    print("Running inference...")

    # 📁 Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    config_path = PROJECT_ROOT / "src" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = PROJECT_ROOT / config["output"]["model_path"]
    vectorizer_path = PROJECT_ROOT / config["output"]["vectorizer_path"]

    # 📦 Load artifacts
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # 🧪 Example input
    text = "Breaking news: shocking discovery about vaccines!"

    # 🔮 Prediction
    pred = predict(text, model, vectorizer)
    label = int(pred[0])

    print(f"\nPrediction: {label}")

    # 🧠 RAG explanation
    print("\nGenerating RAG explanation...")
    rag = RAGExplainer()
    explanation = rag.explain(text, str(label))

    print("\nRAG Explanation:\n")
    print(explanation)

    print("\nInference complete.")


if __name__ == "__main__":
    run_inference()