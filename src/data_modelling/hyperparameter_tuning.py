from sklearn.model_selection import GridSearchCV


def tune_model(model, X_train, y_train, config):

    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=config["training"]["cv_folds"],
        scoring=config["training"]["scoring"],
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_