"""
Tokenization and NLP preprocessing.
"""

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove stopwords from tokens.
    """
    return [t for t in tokens if t not in stop_words]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatize tokens.
    """
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> list[str]:
    """
    Full NLP preprocessing pipeline.
    """

    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    return tokens