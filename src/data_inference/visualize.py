def plot_top_features(contributions, title="Top Features"):
    """
    Plot top contributing features from a model.
    """
    import matplotlib.pyplot as plt

    words, coefs = zip(*contributions)
    plt.figure(figsize=(10,6))
    plt.barh(words, coefs)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()