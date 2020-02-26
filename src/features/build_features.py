# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

TRAIN_SET_PATH = "data/interim/train_set.csv"
TARGET = "0"

train_set = pd.read_csv(TRAIN_SET_PATH)
X_train = train_set.drop(columns=TARGET)
#print(type(X_train).__module__)

#train_set = np.genfromtxt(TRAIN_SET_PATH, delimiter=",")
#X_train = train_set[:, 1:]

def build_df_pipeline(X_train):
    print("Building pipeline.")

    # preprocess numeric features
    print("Preprocessing numeric features.")
    numeric_features = X_train.columns[(X_train.dtypes == "int64") | (X_train.dtypes == "float64")]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # preprocess categorical features
    print("Preprocessing categorical features.")
    categorical_features = X_train.columns[X_train.dtypes == 'object']
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")) 
    ])

    pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features) 
        ]
    )

    joblib.dump(pipeline, "data/processed/pipeline.joblib")
    print("Pipeline was successfully built (saved in ../data/processed).")

#def build_np_pipeline(X_train):
#    scaler = StandardScaler()
#    pipeline = scaler.fit(X_train)
#
#    joblib.dump(pipeline, "data/processed/pipeline.joblib")
#    print("Pipeline was successfully built (saved in ../data/processed).")

if __name__ == "__main__":
    build_df_pipeline(X_train)
