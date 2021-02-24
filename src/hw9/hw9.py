from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
# from catboost import CatBoostClassifier


def gini(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return np.sum(proba * (1 - proba))


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return -np.sum(proba * np.log2(proba))


def gain(left_y, right_y, criterion):
    y = np.concatenate((left_y, right_y))
    return criterion(y) - (criterion(left_y) * len(left_y) + criterion(right_y) * len(right_y)) / len(y)


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : dict
        Словарь, отображающий метки в вероятность того, что объект, попавший в данный лист, принадлжит классу,
         соответствующиему метке.
    """
    def __init__(self, y_arr, n_classes):
        values, counts = np.unique(y_arr, return_counts=True)
        counts = counts / len(y_arr)
        self.dist = np.zeros(n_classes)
        self.dist[values] = counts
        # vc_zip = list(zip(values, counts))
        # self.preds = dict(vc_zip)
        # self.y = max(vc_zip, key=lambda x: x[1])[0]


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
                 min_samples_leaf: int = 1.,
                 max_features: str = "auto"):
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
        self.max_depth = max_depth if max_depth is not None else 1_000_000
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_classes = None
        self.oob_ids_mask = None

    def find_best_split(self, X: np.ndarray, y: np.ndarray, allowed_features: np.ndarray):
        n, m = X.shape
        split_gain, split_value, split_dim = -100, -1, -1
        for col in allowed_features:
            vals = np.unique(X[:, col])

            for val in vals:
                mask = X[:, col] < val
                left = y[mask]
                right = y[~mask]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                g = gain(left, right, self.criterion)
                if g > split_gain:
                    split_value = val
                    split_dim = col
                    split_gain = g

        return split_dim, split_value

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth):
        if depth >= self.max_depth:
            return DecisionTreeLeaf(y, self.n_classes)

        features = np.random.choice(X.shape[1], self.max_features, replace=False)
        split_dim, split_value = self.find_best_split(X, y, features)

        if split_dim == -1:
            return DecisionTreeLeaf(y, self.n_classes)

        mask = X[:, split_dim] < split_value

        left = self.build_tree(X[mask], y[mask], depth + 1)
        right = self.build_tree(X[~mask], y[~mask], depth + 1)

        node = DecisionTreeNode(
            split_dim=split_dim,
            split_value=split_value,
            left=left,
            right=right
        )

        return node

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
        self.n_classes = np.unique(y).shape[0]
        self.max_features = int(np.ceil(np.sqrt(X.shape[1]))) if self.max_features == 'auto' else self.max_features
        self.root = self.build_tree(X, y, 0)

    def walk_down(self, node: Union[DecisionTreeNode, DecisionTreeLeaf], x):
        if type(node) is DecisionTreeLeaf:
            return node.dist
        split_value = node.split_value
        split_dim = node.split_dim
        if x[split_dim] < split_value:
            return self.walk_down(node.left, x)
        else:
            return self.walk_down(node.right, x)

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
        preds = [self.walk_down(self.root, x) for x in X]
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        proba = np.argmax(proba, axis=1)
        return proba


class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators

        self.trees = [DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        ) for _ in range(self.n_estimators)]
        self.n_classes = None
        self.label_encoder = LabelEncoder()
        self.out_bag_inds = np.ndarray

    def fit(self, X, y):
        n, m = X.shape

        self.n_classes = np.unique(y).shape[0]
        y = self.label_encoder.fit_transform(y)

        for i in range(len(self.trees)):
            out_bag_inds = np.zeros(n, dtype='bool')
            ids = np.random.randint(n, size=n)
            ids_uniq = np.unique(ids)
            out_bag_inds[ids_uniq] = True
            self.trees[i].oob_ids_mask = ~out_bag_inds
            self.trees[i].fit(X[ids], y[ids])

    def predict(self, X):
        preds = np.zeros((X.shape[0], self.n_classes))
        for tree in self.trees:
            preds += tree.predict_proba(X)

        preds = np.argmax(preds, axis=1)
        preds = self.label_encoder.inverse_transform(preds)
        return preds
