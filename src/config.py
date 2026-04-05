from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

# 1. Data cleaning configs
DATA_FILE = "WELFake_Dataset.csv"
DATA_CLEANING_INPUT_FILE = os.path.join(RAW_DIR, DATA_FILE)
DATA_CLEANING_OUTPUT_FILE = INTERIM_DIR / f"cleaned_data_{CURRENT_DATE}.parquet"

# 2. Data Preprocessing configs
DATA_PREPROCESSING_INPUT_FILE = DATA_CLEANING_OUTPUT_FILE
DATA_PREPROCESSING_OUTPUT_FILE = INTERIM_DIR / f"preprocessed_data_{CURRENT_DATE}.parquet"

# 3. Data modelling configs
DATA_MODELING_INPUT_FILE = DATA_PREPROCESSING_OUTPUT_FILE
MODEL_PROD_DIR = MODEL_DIR / "production"
MODEL_PROD_DIR.mkdir(parents=True, exist_ok=True)
SRC_MODEL_NAME = "lsvm_model.joblib"
SRC_VECTORIZER_NAME = "vectorizer.joblib"
SRC_VECTORIZER_FILE = MODEL_PROD_DIR / SRC_VECTORIZER_NAME
SRC_MODEL_FILE = MODEL_PROD_DIR / SRC_MODEL_NAME

# 4. Data inference configs
INPUT_FILE_DATA_INFERENCE = RAW_DIR / "..."
MODEL_FILE_INFERENCE = SRC_MODEL_FILE

# Configs for main.py
SRC_DIR = Path(__file__).parent
DATA_CLEANING_MODULE = "src.data_cleaning.run_cleaning"
DATA_PREPROCESSING_MODULE = "src.data_preprocessing.run_preprocessing"
DATA_MODELLING_MODULE = "src.data_modelling.run_modeling"
DATA_INFERENCE_MODULE = "src.data_inference.run_inference"

