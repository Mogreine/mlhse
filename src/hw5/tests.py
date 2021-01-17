import numpy as np
from src.hw5.hw5 import MLPClassifier, Linear, ReLU
import copy
from sklearn.datasets import make_blobs, make_moons
from typing import List, NoReturn


def test1():
    p = MLPClassifier([
        Linear(4, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 2)
    ])

    X = np.random.randn(50, 4)
    y = np.array([(0 if x[0] > x[2] ** 2 or x[3] ** 3 > 0.5 else 1) for x in X])
    p.fit(X, y)
    acc = np.mean(p.predict(X).flatten() == y)
    print("Accuracy", acc)


def test2():
    X, y = make_moons(400, noise=0.075)
    X_test, y_test = make_moons(400, noise=0.075)

    best_acc = 0
    for _ in range(10):
        p = MLPClassifier([
                Linear(X.shape[1], 64),
                ReLU(),
                Linear(64, 64),
                ReLU(),
                Linear(64, 2)
            ],
            epochs=10,
            alpha=0.01)

        p.fit(X, y, batch_size=1)
        best_acc = max(np.mean(p.predict(X_test).flatten() == y_test), best_acc)
    print("Accuracy", best_acc)


def test3():
    X, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    X_test, y_test = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    best_acc = 0
    for _ in range(10):
        p = MLPClassifier([
            Linear(X.shape[1], 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 3)
        ],
            epochs=10,
            alpha=0.01)

        p.fit(X, y, batch_size=1)
        best_acc = max(np.mean(p.predict(X_test).flatten() == y_test), best_acc)
    print("Accuracy", best_acc)


if __name__ == "__main__":
    test2()
    test3()
