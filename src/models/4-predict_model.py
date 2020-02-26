import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
import joblib

from .. import dispatcher

if __name__ == "__main__":

    TEST_SET_PATH = os.environ.get("TEST_SET_PATH")
    FITTED_PIPELINE_PATH = os.environ.get("FITTED_PIPELINE_PATH")
    MODEL = os.environ.get("MODEL")
    TARGET = "0"

    test_set = pd.read_csv(TEST_SET_PATH)
    X_test = test_set.drop(columns=TARGET)
    y_test = test_set[TARGET].copy()

    print("Preparing features for ML algorithm.")
    fitted_pipeline = joblib.load(FITTED_PIPELINE_PATH) 
    X_test_preprocessed = fitted_pipeline.transform(X_test)
    print("Features successfully prepared for ML algorithm.")

    some_data = X_test_preprocessed[:5]
    some_labels = y_test[:5]

    print("Loading prepared model.")
    model = joblib.load(os.path.join("models", f"{MODEL}.pkl"))

    print("Making predictions.")
    predictions = model.predict(some_data)
    print("Predictions successfully made.")

    print("Test model on 5 samples")
    print("Predictions:", predictions)
    print("Labels:", some_labels.values)
    print("Evaluation metrics:\n", np.sqrt(metrics.confusion_matrix(some_labels, model.predict(some_data))))
