import numpy as np


def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    e = y_predicted - y_true
    return 1 / len(e) * e @ e


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    y_true_mean = np.mean(y_true)
    SS_tot = (y_true - y_true_mean) @ (y_true - y_true_mean)
    SS_res = (y_true - y_predicted) @ (y_true - y_predicted)
    return 1 - SS_res / SS_tot


class NormalLR:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        b = np.ones_like(y).reshape(-1, 1)
        _X = np.hstack((X, b))

        self.w = np.linalg.inv(_X.T @ _X) @ _X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        b = np.ones(X.shape[0]).reshape(-1, 1)
        _X = np.hstack((X, b))
        return _X @ self.w


class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.001):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l

    def fit(self, X: np.ndarray, y: np.ndarray):
        b = np.ones_like(y).reshape(-1, 1)
        _X = np.hstack((X, b))
        n = _X.shape[1]
        w = np.zeros(_X.shape[1])
        # w = np.random.uniform(-1 / np.sqrt(n), 1 / np.sqrt(n), size=n)
        self.w = self.optimize(_X, y, w, tol=1e-8, max_iter=self.iterations)

    def stop_criterion(self, x0_grad, x_grad, tol):
        x_grad_norm = np.linalg.norm(x_grad)
        x0_grad_norm = np.linalg.norm(x0_grad)
        rk = x_grad_norm ** 2 / x0_grad_norm ** 2
        return rk < tol

    def optimize(self, X, y, x0, tol=1e-5, max_iter=int(1e4)) -> np.ndarray:
        grad = lambda w: 1 / X.shape[0] * (2 * X.T @ X @ w - 2 * X.T @ y) + self.l * np.sign(w)
        x0_grad = grad(x0)
        x_k = x0.copy()
        for i in range(max_iter):
            xk_grad = grad(x_k)
            p_k = -xk_grad

            alpha = self.alpha

            x_k = x_k + alpha * p_k

            if self.stop_criterion(x0_grad, xk_grad, tol):
                break
        return x_k

    def predict(self, X: np.ndarray):
        b = np.ones(X.shape[0]).reshape(-1, 1)
        _X = np.hstack((X, b))
        return _X @ self.w
