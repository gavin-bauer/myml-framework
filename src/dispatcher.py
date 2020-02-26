import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm

MODELS = {
    "SVC": svm.SVC(kernel="linear"),
    "SGDClassifier": linear_model.SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
}