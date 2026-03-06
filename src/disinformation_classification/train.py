"""
Model training pipeline for the disinformation classifier.
"""

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from .model import build_lstm_model
from .config import MAX_SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, MODEL_PATH


def train_model():
    """
    Train the LSTM disinformation classification model.

    This function assumes that the input data has already
    been tokenized into integer sequences.
    """

    print("Starting model training...")

    # Placeholder training data
    # Replace with your real tokenized dataset
    X_train = np.random.randint(0, 1000, (1000, MAX_SEQUENCE_LENGTH))
    y_train = np.random.randint(0, 2, 1000)

    vocab_size = 1000

    model = build_lstm_model(vocab_size, MAX_SEQUENCE_LENGTH)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    model.save(MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")