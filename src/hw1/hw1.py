import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
from collections import namedtuple
from typing import NoReturn, Tuple, List


class KDTree:

    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.split_feature = 0

        self.leaf = namedtuple('leaf', ('node_type', 'points'))
        self.node = namedtuple('node', ('node_type', 'split_param_name', 'split_param_val', 'lnode', 'rnode'))
        ind = np.array([*range(X.shape[0])]).reshape((-1, 1))
        X = np.hstack([X, ind])

        self.root = self.build(X, leaf_size=leaf_size)

    def build(self, X: np.array, split_feature=0, leaf_size: int = 40):
        if X.shape[0] <= leaf_size:
            return self.leaf('leaf', X.copy())

        split_feature %= X.shape[1] - 1

        arr = X[:, self.split_feature]
        median = np.median(arr)
        median_mask = X[:, self.split_feature] <= median

        xl = X[median_mask]
        xr = X[~median_mask]

        lnode = self.build(xl, split_feature + 1, leaf_size)
        rnode = self.build(xr, split_feature + 1, leaf_size)

        return self.node('node', self.split_feature, median, lnode, rnode)

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        res = []
        for x in X:
            res_x = self.query_one(x, k=k)
            res.append(res_x)
        return res

    def sort_by_dist(self, X, y):
        ind = X[:, -1].reshape((-1, 1))
        X = np.delete(X, obj=X.shape[1] - 1, axis=1)

        # a fast way to calc dist to all points
        dist_to_all_points = np.sum((X.T - y.reshape(-1, 1)) ** 2, axis=0) ** (1 / 2)
        dist_to_all_points = dist_to_all_points.reshape((-1, 1))

        X_w_dists = np.hstack([X, dist_to_all_points, ind])
        X_w_dists = X_w_dists[X_w_dists[:, -2].argsort()]

        return X_w_dists

    @staticmethod
    def dist(x, y):
        return np.linalg.norm(x - y)

    def query_one(self, x: np.array, k=1):
        res = self._query_one(x, node=self.root, k=k)
        return res[:, -1]

    def _query_one(self, x: np.array, node, k=1):
        if node.node_type == 'leaf':
            return self.sort_by_dist(node.points, x)[:k]

        node_order = [node.lnode, node.rnode]
        if x[node.split_param_name] > node.split_param_val:
            node_order.reverse()

        arr1 = self._query_one(x, node_order[0], k)
        # Мы не хотим брать индекс и расстояние до x у точки
        furthest_point = arr1[-1, :-2]

        if KDTree.dist(furthest_point, x) > abs(node.split_param_val - x[node.split_param_name]):
            arr2 = self._query_one(x, node_order[1], k)
            arr = np.vstack([arr1, arr2])
            arr = arr[arr[:, -2].argsort()]
            return arr[:k]
        else:
            return arr1


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
        Вектор бинарных меток, 1 соответствует злокачественной опухоли (M),
        0 --- доброкачественной (B).


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


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    accuracy = np.count_nonzero(y_true == y_pred) / len(y_true)

    classes = np.unique(y_true)
    precision_arr, recall_arr = [], []
    negative, positive = False, True

    for cl in classes:
        y_true_cl = y_true == cl
        y_pred_cl = y_pred == cl

        tp = np.count_nonzero(np.logical_and(y_pred_cl == positive, y_true_cl == positive))
        tn = np.count_nonzero(np.logical_and(y_pred_cl == negative, y_true_cl == negative))
        fp = np.count_nonzero(np.logical_and(y_pred_cl == positive, y_true_cl == negative))
        fn = np.count_nonzero(np.logical_and(y_pred_cl == negative, y_true_cl == positive))

        precision_arr.append(tp / (tp + fp))
        recall_arr.append(tp / (tp + fn))

    return np.array(precision_arr), np.array(recall_arr), accuracy


if __name__ == '__main__':
    cance_X, cancer_y = read_cancer_dataset('data/cancer.csv')
    spam_X, spam_y = read_spam_dataset('data/spam.csv')

    kd_tree = KDTree(cance_X, leaf_size=5)


    print('Done!')
