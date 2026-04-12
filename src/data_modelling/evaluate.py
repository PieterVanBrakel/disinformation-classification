import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(model, X_test, y_test, output_dir: Path):
    """
    Evaluate model and save metrics.

    Returns:
        dict: evaluation results including accuracy, precision, recall, f1,
              full classification report, and raw predictions.
    """

    y_pred = model.predict(X_test)

    report   = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    # Extract weighted averages for top-level logging
    precision = report["weighted avg"]["precision"]
    recall    = report["weighted avg"]["recall"]
    f1        = report["weighted avg"]["f1-score"]

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        }, f, indent=4)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    return {
        "y_pred":    y_pred,
        "report":    report,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }