from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

from src.hw9.hw9 import *


def feature_importance(rfc: RandomForestClassifier, X: np.ndarray, y: np.ndarray):
    n, m = X.shape
    importance = np.zeros((rfc.n_estimators, m))
    for i in range(len(rfc.trees)):
        inds = rfc.trees[i].oob_ids_mask
        X_, y_ = X[inds], y[inds]
        err_oob = np.mean(rfc.predict(X_) != y_)
        for col in range(m):
            X_shuffled = X_.copy()
            np.random.shuffle(X_shuffled[:, col])
            err_oob_col = np.mean(rfc.predict(X_shuffled) != y_)
            importance[i, col] = err_oob_col - err_oob

    importance = np.mean(importance, axis=0)
    return importance


def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    indices = np.argsort(importance)[::-1][:k]
    return np.array(names)[indices]


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = synthetic_dataset(1000)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X, y)
    print("Accuracy:", np.mean(rfc.predict(X) == y))
    print("Importance:", feature_importance(rfc, X, y))
