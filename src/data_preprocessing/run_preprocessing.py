import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nltk

from .text_preprocessing import preprocess_text
from .feature_engineering import add_features

tqdm.pandas()


def setup_nltk():
    import nltk

    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")


def run_preprocessing():

    print("Starting preprocessing pipeline...")

    # 📁 Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    config_path = PROJECT_ROOT / "src" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_path = PROJECT_ROOT / config["data"]["cleaned_path"]
    train_path = PROJECT_ROOT / config["data"]["processed_train_path"]
    test_path = PROJECT_ROOT / config["data"]["processed_test_path"]

    label_col = config["preprocessing"]["label_column"]
    test_size = config["preprocessing"]["test_size"]

    # 📦 Load data
    print("Loading cleaned data...")
    df = pd.read_parquet(input_path)

    # 🧠 Setup NLP
    setup_nltk()

    # 🔤 Text preprocessing
    print("Tokenizing + preprocessing text...")
    df["tokens"] = df["clean_text"].progress_apply(preprocess_text)

    # ⚙️ Feature engineering
    print("Generating features...")
    df = add_features(df)

    # ✂️ Train/test split
    print("Splitting dataset...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=42
    )

    # 💾 Save
    print("Saving processed data...")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()