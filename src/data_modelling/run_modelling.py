import yaml
import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from .vectorizer import build_vectorizer
from .model import build_model
from .hyperparameter_tuning import tune_model
from .visualize import plot_confusion_matrix, plot_class_distribution, plot_f1_scores
from .evaluate import evaluate_model


def run_modeling() -> None:
    """Vectorise, tune, train, evaluate, save artifacts, and log to MLflow."""

    print("Starting modeling pipeline...")

    # 📁 Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    config_path = PROJECT_ROOT / "src" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_path      = PROJECT_ROOT / config["data"]["processed_train_path"]
    test_path       = PROJECT_ROOT / config["data"]["processed_test_path"]
    model_path      = PROJECT_ROOT / config["output"]["model_path"]
    vectorizer_path = PROJECT_ROOT / config["output"]["vectorizer_path"]
    metrics_path    = PROJECT_ROOT / config["output"]["metrics_path"]

    # 🔬 MLflow setup
    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():

        # 📦 Load data
        print("Loading processed data...")
        train_df = pd.read_parquet(train_path, engine="fastparquet")
        test_df  = pd.read_parquet(test_path,  engine="fastparquet")

        X_train = train_df["tokens"].apply(lambda x: " ".join(x))
        y_train = train_df[config["preprocessing"]["label_column"]]
        X_test  = test_df["tokens"].apply(lambda x: " ".join(x))
        y_test  = test_df[config["preprocessing"]["label_column"]]

        # 🔤 Vectorization
        print("Vectorizing text...")
        vectorizer  = build_vectorizer(config)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec  = vectorizer.transform(X_test)

        # 🤖 Model
        print("Training model with hyperparameter tuning...")
        model = build_model()
        best_model, best_params = tune_model(model, X_train_vec, y_train, config)
        print(f"Best params: {best_params}")

        # 📊 Evaluation
        print("Evaluating model...")
        metrics = evaluate_model(best_model, X_test_vec, y_test, metrics_path)
        y_pred  = metrics["y_pred"]

        print("Generating visualizations...")
        plot_confusion_matrix(y_test, y_pred, metrics_path)
        plot_class_distribution(y_test, metrics_path)
        plot_f1_scores(metrics["report"], metrics_path)

        # 🔬 Log parameters
        mlflow.log_param("model_type",    "LinearSVC")
        mlflow.log_param("C",             best_params["C"])
        mlflow.log_param("max_features",  config["model"]["max_features"])
        mlflow.log_param("ngram_range",   str(config["model"]["ngram_range"]))
        mlflow.log_param("cv_folds",      config["training"]["cv_folds"])
        mlflow.log_param("test_size",     config["preprocessing"]["test_size"])

        # 🔬 Log metrics
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1",        metrics["f1"])

        # 🔬 Log artifacts
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact(str(metrics_path / "confusion_matrix.png"))
        mlflow.log_artifact(str(metrics_path / "metrics.json"))

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged. Run ID: {run_id}")

        # 💾 Save artifacts to disk
        print("Saving model and vectorizer...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"  Model saved to:      {model_path}")
        print(f"  Vectorizer saved to: {vectorizer_path}")

    print("Modeling complete.")


if __name__ == "__main__":
    run_modeling()