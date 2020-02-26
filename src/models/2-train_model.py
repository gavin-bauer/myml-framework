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

    some_data = X_train_preprocessed[:5]
    some_labels = y_train[:5]

    model = dispatcher.MODELS[MODEL]
    model_name = model.__class__.__name__
    print(f"Training {model_name}.")
    model.fit(X_train_preprocessed, y_train)

    joblib.dump(fitted_pipeline, "data/processed/fitted_pipeline.joblib")
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"Training of {model_name} was successful (model saved in ../data/processed).")

    print("Test model on 5 samples")
    print("Predictions:", model.predict(some_data))
    print("Labels:", some_labels.values)
    print("Evaluation metrics:\n", metrics.confusion_matrix(some_labels, model.predict(some_data)))