from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(config):
    return TfidfVectorizer(
        max_features=config["model"]["max_features"],
        ngram_range=tuple(config["model"]["ngram_range"])
    )