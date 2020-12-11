import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
import copy
from typing import NoReturn
from src.hw4.hw4 import Perceptron, PerceptronBest


def visualize(X, labels_true, labels_pred, w):
    unique_labels = np.unique(labels_true)
    unique_colors = dict([(l, c) for l, c in zip(unique_labels, [[0.8, 0., 0.], [0., 0., 0.8]])])
    plt.figure(figsize=(9, 9))

    if w[1] == 0:
        plt.plot([X[:, 0].min(), X[:, 0].max()], w[0] / w[2])
    elif w[2] == 0:
        plt.plot(w[0] / w[1], [X[:, 1].min(), X[:, 1].max()])
    else:
        mins, maxs = X.min(axis=0), X.max(axis=0)
        pts = [[mins[0], -mins[0] * w[1] / w[2] - w[0] / w[2]],
               [maxs[0], -maxs[0] * w[1] / w[2] - w[0] / w[2]],
               [-mins[1] * w[2] / w[1] - w[0] / w[1], mins[1]],
               [-maxs[1] * w[2] / w[1] - w[0] / w[1], maxs[1]]]
        pts = [(x, y) for x, y in pts if mins[0] <= x <= maxs[0] and mins[1] <= y <= maxs[1]]
        x, y = list(zip(*pts))
        plt.plot(x, y, c=(0.75, 0.75, 0.75), linestyle="--")

    colors_inner = [unique_colors[l] for l in labels_true]
    colors_outer = [unique_colors[l] for l in labels_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors_inner, edgecolors=colors_outer)
    plt.show()


def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.

    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    h_sym = np.abs(images - images[:, ::-1, :]).mean(axis=(1, 2))
    v_sym = np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2))
    return np.hstack((h_sym.reshape(-1, 1), v_sym.reshape(-1, 1)))


def get_digits(y0=1, y1=5):
    data = datasets.load_digits()
    images, labels = data.images, data.target
    mask = np.logical_or(labels == y0, labels == y1)
    labels = labels[mask]
    images = images[mask]
    images /= np.max(images)
    X = transform_images(images)
    return X, labels


def test():
    X, y = get_digits()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

    c = Perceptron(iterations=10000)
    c.fit(X_train, y_train)
    visualize(X_train, y_train, np.array(c.predict(X_train)), c.w)
    print("Accuracy:", np.mean(c.predict(X_test) == y_test))

    c = PerceptronBest(iterations=10000)
    c.fit(X_train, y_train)
    visualize(X_train, y_train, np.array(c.predict(X_train)), c.w)
    print("Accuracy:", np.mean(c.predict(X_test) == y_test))

    accs = []
    for y0, y1 in [(y0, y1) for y0 in range(9) for y1 in range(y0 + 1, 10)]:
        X, y = get_digits(y0, y1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        c = Perceptron(iterations=2000)
        c.fit(X_train, y_train)
        accs.append(np.mean(c.predict(X_test) == y_test))
    print("Mean accuracy:", np.mean(accs))

    accs = []
    for y0, y1 in [(y0, y1) for y0 in range(9) for y1 in range(y0 + 1, 10)]:
        X, y = get_digits(y0, y1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        c = PerceptronBest(iterations=2000)
        c.fit(X_train, y_train)
        accs.append(np.mean(c.predict(X_test) == y_test))
    print("Mean accuracy:", np.mean(accs))


test()
