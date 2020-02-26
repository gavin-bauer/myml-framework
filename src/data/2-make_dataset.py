# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

FILE_NAMES = ["mnist_train_100.csv", "mnist_test_10.csv"] # order: train.csv, test.csv
TARGET = ""
DATASET_PATH_ROOT = os.path.join("data", "raw")

def make_column_name(df):
    columns = [str(i) for i in range(df.shape[1])]
    return columns

def load_csv(file_name, header=True):
    dataset_path = os.path.join(DATASET_PATH_ROOT, file_name)
    df = pd.read_csv(dataset_path)

    if header == False:
        columns = make_column_name(df)
        df = pd.read_csv(dataset_path, names=columns)

    return df

def split_data(df):
    folds = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    print(f"Splitting raw dataset using {folds}.")

    for train_index, test_index in folds.split(X=df, y=df[TARGET]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]

    train_set.to_csv("data/interim/train_set.csv")
    test_set.to_csv("data/interim/test_set.csv")

    print("Train set size:", len(train_set))
    print("Test set size:", len(test_set))
    
if __name__ == "__main__":
    if len(FILE_NAMES) == 2:
        for i in range(len(FILE_NAMES)):
            if i == 0:
                df = load_csv(FILE_NAMES[0], header=False)
                df.to_csv("data/interim/train_set.csv", index=False)
            elif i == 1:
                df = load_csv(FILE_NAMES[1], header=False)
                df.to_csv("data/interim/test_set.csv", index=False)

    elif len(FILE_NAMES) == 1:
        df = load_csv(FILE_NAMES[0])
        split_data(df)

    print("Train & test sets were successfully split and saved (in ../data/interim).")
