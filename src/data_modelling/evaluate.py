import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(model, X_test, y_test, output_dir: Path):
    """
    Evaluate model and save metrics.

    Returns:
        dict: evaluation results
    """

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full report
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save accuracy separately (nice for dashboards later)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=4)

    print(f"Accuracy: {accuracy:.4f}")

    return {
        "y_pred": y_pred,
        "report": report,
        "accuracy": accuracy
    }