import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
import joblib

from .. import dispatcher

TRAIN_SET_PATH = "data/interim/train_set.csv"
TARGET = "0"
PIPELINE_PATH = "data/processed/pipeline.joblib"

train_set = pd.read_csv(TRAIN_SET_PATH)
X_train = train_set.drop(columns=TARGET)
y_train = train_set[TARGET]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


#train_set = np.genfromtxt(TRAIN_SET_PATH, delimiter=",")
#X_train = train_set[:, 1:]
#y_train = train_set[:, 0]

pipeline = joblib.load(PIPELINE_PATH) 
fitted_pipeline = pipeline.fit(X_train)
X_train_preprocessed = fitted_pipeline.transform(X_train)
print("X processed:\n", X_train_preprocessed.shape)
print(X_train_preprocessed[:5])

def model_benchmark(X_train_preprocessed, y_train):
    # split dataset in cross-validation with this splitter class: 
    folds = model_selection.ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # create table to compare metrics
    table_columns = ["Name", "Parameters", "Train Score Mean", "Test Score Mean",
                    "Test Score STD", "Time"]
    benchmark = pd.DataFrame(columns = table_columns)

    # index through models and save performance to table
    print("Starting model benchmark.")
    row_index = 0
    for key, model in dispatcher.MODELS.items():
        # set name and parameters
        print(f"Evaluating {key}.")
        benchmark.loc[row_index, "Name"] = key
        benchmark.loc[row_index, "Parameters"] = str(model.get_params())

        # score model with cross validation
        cv_results = model_selection.cross_validate(model, X_train_preprocessed, y_train,
                                                    cv=folds, scoring="accuracy",
                                                    return_train_score=True)
        benchmark.loc[row_index, "Time"] = cv_results["fit_time"].mean()
        benchmark.loc[row_index, "Train Score Mean"] = cv_results["train_score"].mean()
        benchmark.loc[row_index, "Test Score Mean"] = cv_results["test_score"].mean()   
        benchmark.loc[row_index, "Test Score STD"] = cv_results["test_score"].std()      

        row_index+=1

    # print and sort table: 
    benchmark.sort_values(by=["Test Score Mean"], ascending=False, inplace=True)
    print("Model benchmark results:")
    print(benchmark)

if __name__ == "__main__":
    model_benchmark(X_train_preprocessed, y_train)