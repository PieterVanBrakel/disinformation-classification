import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, output_dir: Path):
    """
    Plot and save confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def plot_class_distribution(y, output_dir: Path, filename="class_distribution.png"):
    """
    Plot class distribution.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    values = y.value_counts()

    plt.figure()
    plt.bar(values.index.astype(str), values.values)
    plt.title("Class Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def plot_f1_scores(report: dict, output_dir: Path):
    """
    Plot F1 scores per class from classification report.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        label for label in report.keys()
        if label not in ("accuracy", "macro avg", "weighted avg")
    ]

    f1_scores = [report[label]["f1-score"] for label in labels]

    plt.figure()
    plt.bar(labels, f1_scores)
    plt.title("F1 Score per Class")

    plt.tight_layout()
    plt.savefig(output_dir / "f1_scores.png")
    plt.close()