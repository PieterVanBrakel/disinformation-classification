# %%

import pandas as pd
from .text_cleaning import clean_text
import yaml
from tqdm import tqdm
tqdm.pandas()
from pathlib import Path

def run_cleaning():

    print("Starting cleaning pipeline...")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    config_path = PROJECT_ROOT / "src" / "config.yaml"

    # Loading paths
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    input_data = PROJECT_ROOT / config["data"]["raw_path"]
    output_data = PROJECT_ROOT / config["data"]["cleaned_path"]

    # 📦 Load data
    print("Loading cleaned data...")
    df = pd.read_csv(input_data)

    df["text"] = df["text"].fillna("").astype(str)

    # Cleaning text
    df["clean_text"] = df["text"].progress_apply(clean_text)
    output_data.parent.mkdir(parents=True, exist_ok=True)

    # Save the data
    df.to_parquet(output_data, engine = "fastparquet")


if __name__ == "__main__":
    run_cleaning()