def add_features(df):

    df["text_length"] = df["clean_text"].apply(len)

    df["token_count"] = df["tokens"].apply(len)

    df["avg_token_length"] = df["tokens"].apply(
        lambda tokens: sum(len(t) for t in tokens) / len(tokens) if tokens else 0
    )

    return df