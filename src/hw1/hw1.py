import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
from typing import NoReturn, Tuple, List


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).


    """
    data = pd.read_csv(path_to_csv)
    data = data.sample(frac=1).reset_index(drop=True)

    y = data['label']
    data.drop(['label'], axis=1, inplace=True)

    y = y.map({
        'M': 1,
        'B': 0
    })

    return data.to_numpy(), y.to_numpy()


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    data = pd.read_csv(path_to_csv)
    data = data.sample(frac=1).reset_index(drop=True)
    # print(data.head(n=10))

    y = data['label']
    data.drop(['label'], axis=1, inplace=True)

    return data.to_numpy(), y.to_numpy()


def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    train_sz = int(X.shape[0] * ratio)
    # test_sz = X.shape[0] - train_sz

    X_train, X_test = X[:train_sz, :], X[train_sz:, :]
    y_train, y_test = y[:train_sz], y[train_sz:]

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    cance_X, cancer_y = read_cancer_dataset('data/cancer.csv')
    spam_X, spam_y = read_spam_dataset('data/spam.csv')
    print('Done!')
