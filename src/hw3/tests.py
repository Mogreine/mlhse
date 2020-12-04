from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import numpy as np
from src.hw3.hw3 import NormalLR, r2, mse, GradientLR
from sklearn.preprocessing import MinMaxScaler


def read_data(path="boston.csv"):
    dataframe = np.genfromtxt(path, delimiter=",", skip_header=15)
    np.random.seed(42)
    np.random.shuffle(dataframe)
    X = dataframe[:, :-1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = dataframe[:, -1]
    return X, y


def generate_synthetic(size: int, dim=6, noise=0.1):
    X = np.random.randn(size, dim)
    w = np.random.randn(dim + 1)
    noise = noise * np.random.randn(size)
    y = X.dot(w[1:]) + w[0] + noise
    return X, y


def build_plot(X_train, y_train, X_test, y_test):
    xs = np.arange(0.0, 0.01, 0.0002)
    errors = []
    for x in xs:
        regr = GradientLR(0.01, iterations=10000, l=x)
        regr.fit(X_train, y_train)
        metric_mse = mse(y_test, regr.predict(X_test))
        metric_r2 = r2(y_test, regr.predict(X_test))
        errors.append(metric_mse)
        print(metric_mse, metric_r2)

    plt.figure(figsize=(9, 4))
    plt.xlim(xs[0], xs[-1])
    plt.grid()
    plt.plot(xs, errors)
    plt.show()


def test_normal():
    X, y = read_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    build_plot(X_train, y_train, X_val, y_val)

    regr = NormalLR()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_val)
    print(f"MSE: {mse(y_val, y_pred)}, R2: {r2(y_val, y_pred)}")


test_normal()
