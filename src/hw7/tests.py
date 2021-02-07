import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

from src.hw7.hw7 import LinearSVM, KernelSVM, get_polynomial_kernel, get_gaussian_kernel

solvers.options['show_progress'] = False


def visualize(clf, X, y):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_border = (x_max - x_min) / 20 + 1.0e-3
    x_h = (x_max - x_min + 2 * x_border) / 200
    y_border = (y_max - y_min) / 20 + 1.0e-3
    y_h = (y_max - y_min + 2 * y_border) / 200

    cm = plt.cm.Spectral

    xx, yy = np.meshgrid(np.arange(x_min - x_border, x_max + x_border, x_h),
                         np.arange(y_min - y_border, y_max + y_border, y_h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    plt.figure(1, figsize=(8, 8))
    plt.pcolormesh(xx, yy, z_class, cmap=cm, alpha=0.3, shading='gouraud')

    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    plt.contour(xx, yy, z_dist, [0.0], colors='black')
    plt.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points
    y_pred = clf.predict(X)

    ind_support = clf.support.flatten()
    ind_correct = []
    ind_incorrect = []
    for i in range(len(y)):
        # if i in clf.support:
        #     ind_support.append(i)
        if y[i] == y_pred[i]:
            ind_correct.append(i)
        else:
            ind_incorrect.append(i)

    plt.scatter(X[ind_correct, 0], X[ind_correct, 1], c=y[ind_correct], cmap=cm, alpha=1., edgecolor='black',
                linewidth=.8)
    plt.scatter(X[ind_incorrect, 0], X[ind_incorrect, 1], c=y[ind_incorrect], cmap=cm, alpha=1., marker='*',
                s=50, edgecolor='black', linewidth=.8)
    plt.scatter(X[ind_support, 0], X[ind_support, 1], c=y[ind_support], cmap=cm, alpha=1., edgecolor='yellow',
                linewidths=1.,
                s=40)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.tight_layout()


def generate_dataset(moons=False):
    if moons:
        X, y = make_moons(1000, noise=0.075, random_state=42)
        return X, 2 * y - 1
    X, y = make_blobs(1000, 2, centers=[[0, 0], [-4, 2], [3.5, -2.0], [3.5, 3.5]], random_state=42)
    y = 2 * (y % 2) - 1
    return X, y


def test1():
    X, y = generate_dataset(True)
    svm = LinearSVM(1)
    svm.fit(X, y)
    visualize(svm, X, y)
    plt.show()

    X, y = generate_dataset(False)
    svm = LinearSVM(1)
    svm.fit(X, y)
    visualize(svm, X, y)
    plt.show()


def test2():
    # X, y = generate_dataset(True)
    # svm = KernelSVM(1, kernel=get_polynomial_kernel(1, 3))
    # svm.fit(X, y)
    # visualize(svm, X, y)
    # plt.show()
    #
    # X, y = generate_dataset(False)
    # svm = KernelSVM(1, kernel=get_polynomial_kernel(1, 3))
    # svm.fit(X, y)
    # visualize(svm, X, y)
    # plt.show()

    X, y = generate_dataset(True)
    svm = KernelSVM(1, kernel=get_gaussian_kernel(0.4))
    svm.fit(X, y)
    visualize(svm, X, y)
    plt.show()

    X, y = generate_dataset(False)
    svm = KernelSVM(1, kernel=get_gaussian_kernel(0.4))
    svm.fit(X, y)
    visualize(svm, X, y)
    plt.show()

    X, y = generate_dataset(True)
    svm = KernelSVM(1, kernel=get_polynomial_kernel(1, 3))
    svm.fit(X, y)
    visualize(svm, X, y)
    plt.show()


if __name__ == "__main__":
    test2()
