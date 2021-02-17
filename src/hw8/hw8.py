from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    values, counts = np.unique(x, return_counts=True)
    counts /= len(x)
    return np.sum(counts * (1 - counts))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    values, counts = np.unique(x, return_counts=True)
    counts /= len(x)
    return -np.sum(counts * np.log2(counts + 1e-16))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    k1 = len(left_y) / (len(left_y) + len(right_y))
    k2 = len(right_y) / (len(left_y) + len(right_y))
    g = 1 - k1 * criterion(left_y) - k2 * criterion(right_y)
    return g


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : dict
        Словарь, отображающий метки в вероятность того, что объект, попавший в данный лист, принадлжит классу,
         соответствующиему метке.
    """
    def __init__(self, y_arr):
        self.y = None
        values, counts = np.unique(y_arr, return_counts=True)
        counts /= len(y_arr)
        self.y = dict(zip(values, counts))


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """
    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        criteria = {
            'gini': gini,
            'entropy': entropy
        }
        self.criterion = criteria[criterion]
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        n, m = X.shape
        split_gain, split_value, split_dim = -1, 0, 0
        for col in range(m):
            vals = np.unique(X[:, col])

            for val in vals:
                mask = X[:, col] < val
                left = y[mask]
                right = y[~mask]
                if len(left) == 0 or len(right) == 0:
                    continue
                g = gain(left, right, self.criterion)
                if g > split_gain:
                    split_value = val
                    split_dim = col

        return split_dim, split_value

    def build_tree(self, node: DecisionTreeNode, X: np.ndarray, y: np.ndarray):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        X_ = X.copy()
        # X_ = np.sort(X_, axis=0)


    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """

        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
