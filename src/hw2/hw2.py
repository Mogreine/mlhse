from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn
from sklearn.neighbors import KDTree


class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """

        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.

        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

    def _calc_dist(self, X, points):
        return np.array([np.sum((X - y) ** 2, axis=1) ** (1 / 2) for y in points])

    def _init_centroids(self, n, X) -> np.ndarray:
        if self.init == 'sample':
            ind = np.random.choice(X.shape[0], size=n)
            return X[ind]
        if self.init == 'random':
            mins = np.amin(X, axis=0)
            maxs = np.amax(X, axis=0)
            res = np.random.uniform(mins, maxs, (n, X.shape[1]))
            return res
        if self.init == 'k-means++':
            ind = np.random.randint(X.shape[0], size=n)
            centroids = np.array(X[ind])
            for i in range(1, n):
                dists = self._calc_dist(X, centroids)
                dists = np.amin(dists, axis=0) ** 2
                high = np.sum(dists)
                prob = np.random.uniform(high, size=1)

                ind = 0
                sum = 0
                for j in range(X.shape[0]):
                    sum += dists[i]
                    if prob <= sum:
                        ind = j
                        break
                centroids = np.vstack((centroids, X[ind]))
            return centroids

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать
            параметры X и y, даже если y не используется).

        """
        centroids = self._init_centroids(self.n_clusters, X)
        y = np.zeros(X.shape[0])
        for _ in range(self.max_iter):
            y = np.argmin(self._calc_dist(X, centroids), axis=0)
            centroids = np.array([np.mean(X[y == cluster], axis=0) for cluster in range(self.n_clusters)])
        self.centroids = centroids

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера,
        к которому относится данный элемент.

        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.

        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров
            (по одному индексу для каждого элемента из X).

        """
        y = np.argmin(self._calc_dist(X, self.centroids), axis=0)
        return y


class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        metrics = {
            'euclidean': lambda x, y: np.linalg.norm(x - y),
            'manhattan': lambda x, y: np.sum(np.abs(x - y)),
            'chebyshev': lambda x, y: np.max(np.abs(x - y))
        }
        self.dist = metrics[metric]
        self.metric = metric
        self.neighbours = {}

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        kd = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        y = np.ones(X.shape[0], dtype=int) * -1
        g = kd.query_radius(X, self.eps,
                            return_distance=False,
                            count_only=False)

        def dfs(x):
            nb_inds = g[i]
            for v in nb_inds:
                v_nb = g[i]
                if len(v_nb) - 1 <= self.min_samples:
                    y[v] = y[x]
                    continue
                if y[v] == -1:
                    y[v] = y[x]
                    dfs(v)

        cl = 0
        for i in range(len(y)):
            if len(g[i]) > self.min_samples and y[i] == -1:
                y[i] = cl
                cl += 1
                dfs(i)
        return y


class AgglomertiveClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.linkage = linkage
        self.n_clusters = n_clusters
        dd = {
            'average': lambda a, b: (a + b) / 2,
            'single': min,
            'complete': max
        }
        self.metric = dd[linkage]

    def _calc_dist(self, cl1, cl2):
        return np.array([np.sum((cl1 - y) ** 2, axis=1) ** (1 / 2) for y in cl2])

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        y = np.arange(X.shape[0])
        i = len(y)
        metric_matrix = self._calc_dist(X, X)

        for i in range(len(y)):
            metric_matrix[i, i] = 1e9

        while i != self.n_clusters:
            clusters = np.unique(y)
            cl1, cl2 = np.unravel_index(np.argmin(metric_matrix, axis=None), metric_matrix.shape)
            for cl in clusters:
                if cl == cl1:
                    continue
                metric_matrix[cl1, cl] = self.metric(metric_matrix[cl1, cl], metric_matrix[cl2, cl])
                metric_matrix[cl, cl1] = metric_matrix[cl1, cl]

            y[y == cl2] = cl1
            metric_matrix[cl2] = 1e9
            metric_matrix[:, cl2] = 1e9

            i -= 1

        clusters = np.sort(np.unique(y))
        for i in range(len(clusters)):
            y[y == clusters[i]] = i

        return y
