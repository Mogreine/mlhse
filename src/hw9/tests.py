from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

from src.hw9.hw9 import *


def feature_importance(rfc):
    raise NotImplementedError()


def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = synthetic_dataset(1000)
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X, y)
    print("Accuracy:", np.mean(rfc.predict(X) == y))
    # print("Importance:", feature_importance(rfc))
