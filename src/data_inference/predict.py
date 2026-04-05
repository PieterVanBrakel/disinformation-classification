def predict(texts, model, vectorizer):
    """
    Predict labels for input texts.
    """

    if isinstance(texts, str):
        texts = [texts]

    X = vectorizer.transform(texts)
    return model.predict(X)