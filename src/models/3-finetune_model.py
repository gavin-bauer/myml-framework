import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import joblib

from .. import dispatcher

if __name__ == "__main__":

    MODEL = os.environ.get("MODEL")
    TRAIN_SET_PATH = os.environ.get("TRAIN_SET_PATH")
    TARGET = "0"
    PIPELINE_PATH = "data/processed/pipeline.joblib"

    train_set = pd.read_csv(TRAIN_SET_PATH)
    X_train = train_set.drop(columns=TARGET)
    y_train = train_set[TARGET].copy()

    print("Preparing features for ML algorithm.")
    pipeline = joblib.load(PIPELINE_PATH) 
    fitted_pipeline = pipeline.fit(X_train)
    X_train_preprocessed = fitted_pipeline.transform(X_train)
    print("Features successfully prepared for ML algorithm.")

    linreg_param_grid = [
        {"fit_intercept": [True, False], "normalize": [True, False]}
    ]

    rf_param_grid = [
        {"n_estimators": [120, 300, 500, 800, 1200],
         "max_depth": [5, 8, 15, 25, 30, None],
         "min_samples_split": [2, 5, 10, 15, 100],
         "min_samples_leaf": [1, 2, 5, 10], 
         "max_features": ["log2", "sqrt", None]}
    ]

    svm_param_grid = [
        {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]},
        {"gamma": "auto"}
    ]

    sgd_param_grid = [
        {"max_iter": [1, 5, 10]},
        {"tol": [1e-3, -np.infty]}
    ]

    mapping = {"LinearRegression": linreg_param_grid, 
           "RandomForestRegressor": rf_param_grid,
           "SVR": svm_param_grid,
           "SGDClassifier": sgd_param_grid}
    param_grid = mapping.get(MODEL)

    model = dispatcher.MODELS[MODEL]
    model_name = model.__class__.__name__

    print(f"\nSearching for best parameters for {model_name}.")
    grid_search = model_selection.GridSearchCV(model, param_grid,
        cv=5, scoring="accuracy", return_train_score=True)

    grid_search.fit(X_train_preprocessed, y_train)

    print("\nBest estimator:")
    print(grid_search.best_estimator_)

    cv_results = grid_search.cv_results_

    print("\nCV results:")
    for score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print(score, params)
