"""
Model architecture definitions for the disinformation classifier.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Dropout


def build_lstm_model(vocab_size: int, max_length: int):
    """
    Build and compile an LSTM model for text classification.

    Parameters
    ----------
    vocab_size : int
        Size of the tokenizer vocabulary.

    max_length : int
        Maximum sequence length.

    Returns
    -------
    keras.Model
        Compiled LSTM model.
    """

    model = Sequential()

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=100,
            input_length=max_length
        )
    )

    model.add(LSTM(128))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model