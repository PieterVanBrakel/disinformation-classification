"""
Prediction module for the disinformation classifier.

This module loads the trained model and preprocessing objects and provides
a single prediction interface that can be reused by:

- CLI
- FastAPI
- notebooks
- batch pipelines
"""

from pathlib import Path
import joblib
import numpy as np

from disinformation_classification.preprocessing import preprocess_text

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path("models/lstm_model.joblib")
TOKENIZER_PATH = Path("models/tokenizer.joblib")


# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------

model = joblib.load(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)


# ---------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------

def predict(text: str) -> dict:
    """
    Predict whether a text contains disinformation.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    dict
        Prediction result containing label and probability.
    """

    # preprocess input
    clean_text = preprocess_text(text)

    # convert to sequence
    sequence = tokenizer.texts_to_sequences([clean_text])

    # pad sequence
    padded = np.array(sequence)

    # model prediction
    prob = model.predict(padded)[0]

    label = int(prob > 0.5)

    return {
        "label": label,
        "probability": float(prob)
    }