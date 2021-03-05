from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import spacy
from nltk.stem.snowball import SnowballStemmer
from typing import NoReturn
import string


class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha
        self.bins = 10
        self.probs = np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """
        self.enc = LabelEncoder()
        y = self.enc.fit_transform(y)

        n_classes = np.amax(y) + 1
        n_features = X.shape[1]

        probs = np.zeros((n_classes, n_features, self.bins), dtype=float)
        for cl in range(n_classes):
            X_ = X[y == cl]
            n_samples = X_.shape[0]
            for feature in range(n_features):
                # hist = np.histogram(X_[:, feature], bins=self.bins, range=(0, self.bins))
                for val in range(self.bins):
                    mask = X_[:, feature] == val
                    probs[cl, feature, val] = np.count_nonzero(mask)

                probs[cl, feature, -1] += np.count_nonzero(X_[:, feature] >= self.bins)

            probs[cl] = (probs[cl] + self.alpha) / (n_samples + self.bins * self.alpha)

        self.probs = np.log(probs + 1e-12)

    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return self.enc.inverse_transform(np.argmax(self.log_proba(X), axis=1))

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу.
            Матрица размера (X.shape[0], n_classes)
        """
        res = []

        X_ = np.minimum(X, self.bins - 1)
        for x in X_:
            fr = np.arange(len(x))
            r = self.probs[:, fr, x]
            r = np.sum(r, axis=1)
            res.append(r)

        return np.array(res)


class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        self.sentences = X
        self.voc_size = voc_limit
        self.fit()

    def fit(self):
        wrds, counts = np.unique(
            [wrd.lower().translate(str.maketrans('', '', string.punctuation)) for sen in self.sentences for wrd in
             sen.split()],
            return_counts=True)
        arr = sorted(zip(counts[1:], wrds[1:]), reverse=True)
        _, voc = zip(*arr)
        self.voc_map = dict(zip(
            voc[:self.voc_size],
            range(self.voc_size))
        )
        self.voc = set(voc[:self.voc_size])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            который необходимо векторизовать.

        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        res = np.zeros((X.shape[0], self.voc_size))

        for sen, i in zip(X, range(X.shape[0])):
            wrds, counts = np.unique([wrd.translate(str.maketrans('', '', string.punctuation)) for wrd in sen.split()],
                                     return_counts=True)
            for w, c in zip(wrds, counts):
                if w in self.voc:
                    res[i][self.voc_map[w]] = c

        return res.astype(int)


class BowStem:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        self.sentences = X
        self.voc_size = voc_limit
        self.stemmer = SnowballStemmer('english')
        self.fit()

    def fit(self):
        wrds = [wrd.lower().translate(str.maketrans('', '', string.punctuation)) for sen in self.sentences for wrd in
                sen.split()]
        wrds = [self.stemmer.stem(wrd) for wrd in wrds]
        wrds, counts = np.unique(wrds, return_counts=True)
        arr = sorted(zip(counts[1:], wrds[1:]), reverse=True)
        _, voc = zip(*arr)
        self.voc_map = dict(zip(
            voc[:self.voc_size],
            range(self.voc_size))
        )
        self.voc = set(voc[:self.voc_size])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            который необходимо векторизовать.

        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        res = np.zeros((X.shape[0], self.voc_size))

        for sen, i in zip(X, range(X.shape[0])):
            wrds, counts = np.unique([wrd.translate(str.maketrans('', '', string.punctuation)) for wrd in sen.split()],
                                     return_counts=True)
            for w, c in zip(wrds, counts):
                if w in self.voc:
                    res[i][self.voc_map[w]] = c

        return res.astype(int)
