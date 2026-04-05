def explain_prediction(text, model, vectorizer, top_n=10):
    """
    Show most important words contributing to prediction.
    """

    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    coef = model.coef_[0]

    indices = X.nonzero()[1]

    contributions = [(feature_names[i], coef[i]) for i in indices]

    contributions = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return contributions[:top_n]