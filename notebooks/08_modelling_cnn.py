# %%

# Standard library
from pathlib import Path

# Third-party libraries
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten

from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# %%

# Variables

# Data
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # fallback for notebooks or interactive environments
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR_INTERIM: Path = PROJECT_ROOT / "data" / "interim"
MODEL_DIR_EXPERIMENT: Path = PROJECT_ROOT / "models" / "experiment"
MODEL_FILE: Path = MODEL_DIR_EXPERIMENT / "best_cnn_model.pkl"
TOKENIZER_FILE: Path = MODEL_DIR_EXPERIMENT / "tokenizer.joblib"

MAX_WORDS = 50000   # vocabulary size
MAX_LEN = 300       # max tokens per document
MAX_FEATURES = 512 # Max words in the vectors

param_grid = {
    "batch_size": [32, 64],
    "epochs": [5, 10],
    "model__filters": [64, 128],
    "model__kernel_size": [3, 5],
    "model__dropout_rate": [0.3, 0.5]
}
# %%

# Load preprocessed tokenized data
X_train = joblib.load(DATA_DIR_INTERIM / "X_train.pkl")  # tokenized train texts
y_train = joblib.load(DATA_DIR_INTERIM / "y_train.pkl")  # train labels

X_test = joblib.load(DATA_DIR_INTERIM / "X_test.pkl")    # tokenized test texts
y_test = joblib.load(DATA_DIR_INTERIM / "y_test.pkl")    # test labels

print(f"Train size: {len(X_train)}")  
print(f"Test size:  {len(X_test)}")

# %%

# 1: Convert token lists back to strings (Tokenizer expects text)
X_train_text = X_train.apply(lambda tokens: " ".join(tokens))
X_test_text  = X_test.apply(lambda tokens: " ".join(tokens))

# 2: Build a Tokenizer (or use one saved earlier)
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")

# 3: Fit the tokenizer only on training data to avoid data leakage.
tokenizer.fit_on_texts(X_train_text)

# 4: transform the training and test data from texts to numbers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

MAX_LEN = 200

# 5: Pad sequences to transform the shape for the X_values for the CNN

# 5.1 For the training data
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# For the test data
X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# 6: Transform the y_values for the CNN

y_train = y_train.values if hasattr(y_train, "values") else y_train
y_test = y_test.values if hasattr(y_test, "values") else y_test

# 7: Create a model-building function for GridSearch

def create_cnn_model(
    filters=128,
    kernel_size=5,
    dense_units=64,
    dropout_rate=0.5,
    optimizer="adam"
):
    model = Sequential(name="cnn_text_classifier")

    # Explicit input layer (THIS FIXES THE ERROR)
    model.add(Input(shape=(MAX_LEN,)))

    model.add(Embedding(input_dim=MAX_FEATURES, output_dim=128))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# 8: Wrap with Kerasclassfier

model = KerasClassifier(
    model=create_cnn_model,
    verbose=0
)

# 9 Run the model with gridsearch

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train_pad, y_train)

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
