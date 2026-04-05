"""
Text cleaning utilities.
"""

import string


def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.
    """
    return "".join(c for c in text if c not in string.punctuation)


def clean_text(text: str) -> str:
    """
    Apply basic cleaning steps.
    """

    text = remove_punctuation(text)
    text = lowercase_text(text)

    return text