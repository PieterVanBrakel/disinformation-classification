
"""
RNN with LSTM Training

This file trains and evaluates an LSTM-based text classifier using
preprocessed train/test splits.

Pipeline:
1) Load preprocessed tokenized data
2) Create validation split from train
3) Tokenize + pad sequences
4) Hyperparameter tuning with GridSearchCV (SciKeras wrapper)
5) Evaluate on test set
6) Save best model + tokenizer

To do:

1) Optimize layers
2) Add more text strings

"""

# %%

# Quick test of the .venv
import sys
print(sys.executable)

# Standard library
from pathlib import Path
from __future__ import annotations  # for modern type hints

# Third-party libraries
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# SciKeras wrapper
from scikeras.wrappers import KerasClassifier

# %%

# Config
RANDOM_STATE = 42
SEED = 7

MAX_FEATURES = 512  # maximum number of words to keep in tokenizer
MAX_LEN = 200         # maximum sequence length after padding
MAX_WORDS = 5000

PARAM_GRID = {
    "batch_size": [32, 64],
    "epochs": [5, 10],
    #"model__embedding_dim": [64, 128],
    #"model__lstm_units": [64, 128],
}

# Data
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # fallback for notebooks or interactive environments
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR_INTERIM: Path = PROJECT_ROOT / "data" / "interim"
MODEL_DIR_EXPERIMENT: Path = PROJECT_ROOT / "models" / "experiment"
MODEL_FILE: Path = MODEL_DIR_EXPERIMENT / "best_rnn_lstm_model.pkl"
TOKENIZER_FILE: Path = MODEL_DIR_EXPERIMENT / "tokenizer.joblib"

# Reproducibility
np.random.seed(SEED)        # seed for numpy operations
tf.random.set_seed(SEED)    # seed for tensorflow operations

# %%

# Convert token lists to strings for Tokenizer
def tokens_to_text(tokens) -> str:
    """
    Convert a list of tokens into a whitespace-separated string.
    
    Args:
        tokens (list[str]): List of token strings.
    
    Returns:
        str: Tokens joined by a single space.
    """
    return " ".join(tokens)

def build_lstm_model():
        model = Sequential(name="lstm_model")

        model.add(tf.keras.Input(shape=(MAX_LEN,), name="input_layer"))
        model.add(layer = Embedding(
            input_dim=MAX_FEATURES, 
            output_dim = 128,
            name = "1st_layer"))
        model.add(layer = LSTM(
            units = 128,
            dropout = 0.2,
            recurrent_dropout = 0.2,
            name = "2nd_layer"))
        model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
        model.add(layer = Dense(units = 64,  activation = 'relu', name = '4th_layer'))
        model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
        model.add(layer = Dense(units = 1,  activation = 'sigmoid', name = 'output_layer'))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"])
        return model
    

# %%

# Load preprocessed tokenized data
X_train = joblib.load(DATA_DIR_INTERIM / "X_train.pkl")  # tokenized train texts
y_train = joblib.load(DATA_DIR_INTERIM / "y_train.pkl")  # train labels

X_test = joblib.load(DATA_DIR_INTERIM / "X_test.pkl")    # tokenized test texts
y_test = joblib.load(DATA_DIR_INTERIM / "y_test.pkl")    # test labels

print(f"Train size: {len(X_train)}")  
print(f"Test size:  {len(X_test)}")

# Convert all splits
X_train_text = X_train.apply(tokens_to_text)
X_test_text  = X_test.apply(tokens_to_text)

# Tokenizer + sequences
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")  # create tokenizer
tokenizer.fit_on_texts(X_train_text)  # fit on training data only

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq  = tokenizer.texts_to_sequences(X_test_text)

# Pad sequences to uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

keras_clf = KerasClassifier(
    model=build_lstm_model,
    verbose=0
)

grid = GridSearchCV(
    estimator=keras_clf,
    param_grid=PARAM_GRID,
    scoring="accuracy",
    cv=3,             # cross-validation folds
    n_jobs=-1,        # parallelize
    verbose=2
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,                  # stop if no improvement after 2 epochs
    restore_best_weights=True
)

grid.fit(
    X_train_pad, 
    y_train,
    callbacks=[early_stop])


print("\nBest CV score:", grid.best_score_)
print("Best params:", grid.best_params_)

# Evaluate best model on test set
best_model = grid.best_estimator_

# Use padded test data
y_pred = best_model.predict(X_test_pad)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%

# Save best model + tokenizer
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

joblib.dump(best_model, MODEL_FILE)
joblib.dump(tokenizer, TOKENIZER_FILE)

print(f"\nSaved model to: {MODEL_FILE}")
print(f"Saved tokenizer to: {TOKENIZER_FILE}")
