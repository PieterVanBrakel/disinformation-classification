"""
Configuration settings for the disinformation classification project.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"

# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lstm_model.keras"

# Training parameters
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10