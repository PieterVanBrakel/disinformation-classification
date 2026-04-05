import joblib
import yaml
from pathlib import Path

from .predict import predict
from .explain import explain_prediction
from .visualize import plot_top_features


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
    print(f"\nPrediction: {pred[0]}")

    # 🧠 Explanation
    contributions = explain_prediction(text, model, vectorizer)

    print("\nTop contributing words:")
    for word, score in contributions:
        print(f"{word}: {score:.4f}")

    # 📊 Visualization
    output_dir = PROJECT_ROOT / "models" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_top_features(contributions, output_dir / "feature_importance.png")

    print("\nInference complete.")


if __name__ == "__main__":
    run_inference()