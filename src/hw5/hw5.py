import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
from typing import List, NoReturn


class Module:
    """
    Абстрактный класс. Его менять не нужно.
    """

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features)
        self.b = np.random.randn(out_features)
        self.dW = np.ndarray
        self.db = np.ndarray
        self.X = np.ndarray

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.X = X.copy()
        return X @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        grad : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.dW = self.X.T @ grad
        self.db = np.sum(grad, axis=0)
        dX = grad @ self.W.T
        return dX

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * self.dW
        self.b -= alpha * self.db


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU.
    """

    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.X = X.copy()
        self.X[self.X < 0] = 0
        return self.X

    def backward(self, grad) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        grad : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        dX = self.X.copy()
        dX[dX > 0] = 1
        return dX * grad


class Softmax(Module):
    """
    Слой, соответствующий функции активации Softmax.
    """

    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Softmax(x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        def softmax_naive(X):
            res = np.exp(X)
            sums = np.sum(res, axis=1) + 1e-16
            res /= sums
            return res

        def softmax_stable(X):
            shiftx = X - np.max(X)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        res = softmax_stable(X)
        self.S = res.copy()

        return res

    def backward(self, Y) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки + cross-entropy.

        Parameters
        ----------
        Y : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.grad = Y - self.S
        return self.grad


class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и
            описывающий слои нейронной сети.
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обученияю
        alpha : float
            Cкорость обучения.
        """
        self.modules = copy.deepcopy(modules)
        self.modules.append(Softmax())
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size=32) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу, а используя батчи.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        batch_size : int
            Размер батча.
        """
        samples = X.shape[0]
        features = X.shape[1]
        batches = samples // batch_size + (samples % batch_size != 0)

        # y = y.reshape(-1, 1)
        classes = np.amax(y) + 1
        Y = np.zeros(shape=(samples, classes))
        Y[np.arange(samples), y] = 1

        for epoch in range(self.epochs):
            for batch_number in range(batches):
                input = X[batch_number:batch_number + batch_size]

                # forward cycle
                for layer in self.modules:
                    input = layer.forward(input)

                grad = Y[batch_number:batch_number + batch_size]
                self.modules.reverse()
                # backward cycle
                for layer in self.modules:
                     grad = layer.backward(grad)
                self.modules.reverse()

                # update cycle
                for layer in self.modules:
                    layer.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        input = X.copy()
        # forward cycle
        for layer in self.modules:
            input = layer.forward(input)
        return input

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
