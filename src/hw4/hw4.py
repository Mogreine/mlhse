import numpy as np
from typing import NoReturn


class Perceptron:
    def __init__(self, iterations: int = 1000):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = np.ndarray
        self.max_iter = iterations

    # works only for our datasets
    def transform_labels(self, y):
        u = np.unique(y)
        if u[0] == 0:
            return y * 2 - 1, lambda x: (x + 1) / 2
        else:
            return (y - 1) / 2 - 1, lambda x: (x + 1) * 2 + 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон.
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        bias = np.ones_like(y).reshape(-1, 1)
        X = np.hstack((bias, X))
        self.w = np.zeros(X.shape[1])
        y, transform_back = self.transform_labels(y)
        self.transform_back = transform_back

        for _ in range(self.max_iter):
            yk = np.sign(X @ self.w)
            mask = yk != y
            elem = X[mask][0]
            y_true = y[mask][0]
            self.w += y_true * elem

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        bias = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.hstack((bias, X))
        y = np.sign(X @ self.w)
        return self.transform_back(y)


class PerceptronBest:

    def __init__(self, iterations: int = 1000):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.best_acc = 0
        self.w = np.ndarray
        self.max_iter = iterations

    # works only for our datasets
    def transform_labels(self, y):
        u = np.unique(y)
        if u[0] == 0:
            return y * 2 - 1, lambda x: (x + 1) / 2
        else:
            return (y - 1) / 2 - 1, lambda x: (x + 1) * 2 + 1

    def acc(self, y_pred, y_true):
        return np.count_nonzero(y_true == y_pred) / len(y_true)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса,
        при которых значение accuracy было наибольшим.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        bias = np.ones_like(y).reshape(-1, 1)
        X = np.hstack((bias, X))
        self.w = np.zeros(X.shape[1])
        y, transform_back = self.transform_labels(y)
        self.transform_back = transform_back
        self.w_best = self.w.copy()

        for _ in range(self.max_iter):
            yk = np.sign(X @ self.w)

            curr_acc = self.acc(yk, y)
            if curr_acc > self.best_acc:
                self.best_acc = curr_acc
                self.w_best = self.w.copy()

            mask = yk != y
            elem = X[mask][0]
            y_true = y[mask][0]
            self.w += y_true * elem

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        bias = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.hstack((bias, X))
        y = np.sign(X @ self.w_best)
        return self.transform_back(y)
