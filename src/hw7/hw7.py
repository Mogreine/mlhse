import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable


class LinearSVM:
    def __init__(self, C: float):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.

        """
        self.C = C
        self.support = None
        self.x = np.ndarray

    def find_support(self, X):
        res = X @ self.w + self.w0
        self.support = abs(abs(res) - 1) < 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        y = y.reshape(-1, 1)
        n, m = X.shape
        w = np.random.randn(m + 1)
        ksi = np.random.randn(n)

        x = np.hstack([w, ksi])
        P = np.block([
            [np.eye(m), np.zeros(shape=(m, n + 1))],
            [np.zeros(shape=(n + 1, m)), np.zeros(shape=(n + 1, n + 1))]
        ])
        q = np.hstack([np.zeros(m + 1), self.C * np.ones(n)])
        h = np.hstack([np.zeros(n), -np.ones(n)])
        G = np.block([
            [np.zeros((n, m)), np.zeros((n, 1)), -np.eye(n)],
            [-y * X,           -y,               -np.eye(n)]
        ])
        res = solvers.qp(matrix(P),
                         matrix(q),
                         matrix(G),
                         matrix(h))
        self.w = res['x'][:m]
        self.w0 = res['x'][m]

        # support_ind = res['x'][m:] > 1e-6
        # self.support = np.array(res['z'])[n:] > 1e-6

        self.find_support(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        return X @ self.w + self.w0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))


def get_polynomial_kernel(c=1, power=2):
    """Возвращает полиномиальное ядро с заданной константой и степенью"""
    return lambda u, v: (c + u @ v) ** power


def get_gaussian_kernel(sigma=1.):
    """Возвращает ядро Гаусса с заданным коэффицинтом сигма"""
    def func(X, v):
        if len(X.shape) == 1:
            return np.exp(-sigma * (X - v) @ (X - v))
        X = X.copy()
        X -= v
        X = np.linalg.norm(X, axis=1) ** 2
        return np.exp(-sigma * X)

    return func


class KernelSVM:
    def __init__(self, C: float, kernel: Callable = lambda u, v: u @ v):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.

        """
        self.C = C
        self.kernel = kernel
        self.support = []
        self.a = np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        n, m = X.shape
        kerP = np.array([[self.kernel(X[i], X[j]) for j in range(n)] for i in range(n)])
        P = np.outer(y, y) * kerP
        q = -np.ones(n).reshape(-1, 1)
        G = np.block([
            [np.eye(n)],
            [-np.eye(n)]
        ])
        h = np.hstack([self.C * np.ones(n), np.zeros(n)])
        b = 0.
        A = y.reshape(1, -1,).astype('float64')

        qp_args = [P, q, G, h, A, b]
        qp_args = map(matrix, qp_args)
        P, q, G, h, A, b = qp_args
        res = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        self.alpha = np.array(res['x'])

        self.w0 = 0
        for i in range(n):
            tmp = 0
            for j in range(n):
                tmp += self.alpha[j] * y[j] * self.kernel(X[i], X[j])
            self.w0 += y[i] - tmp
        self.w0 /= n

        # just a filler
        self.support = self.alpha > 100

        self.X = X.copy()
        self.y = y.copy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        res = np.zeros(X.shape[0]) + self.w0
        for i in range(self.X.shape[0]):
            res += self.alpha[i] * self.y[i] * self.kernel(X, self.X[i])
        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))
