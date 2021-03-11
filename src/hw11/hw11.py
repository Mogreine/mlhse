import numpy as np


def cyclic_distance(points, dist):
    res = 0
    for i in range(len(points) - 1):
        res += dist(points[i], points[i + 1])

    return res


def l2_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))


def l1_distance(p1, p2):
    d = p1 - p2
    return np.sqrt(d @ d)


class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist  # Do not change

    def optimize(self, X):
        res = self.optimize_explain(X)
        return res[-1]

    def optimize_explain(self, X: np.ndarray):
        n = X.shape[0]
        perm = np.random.permutation(n)
        X = X[perm]
        dist_arr, curr_loss = self.calc_dist_array(X, self.dist)
        res = [perm]

        for _ in range(self.max_iterations):
            inds, new_loss, new_dist_arr = self.find_best_pair(X, dist_arr, curr_loss)

            if new_loss >= curr_loss:
                break

            curr_loss = new_loss

            new_perm = perm.copy()

            X[inds[0]:inds[1] + 1] = X[inds[1]:inds[0] - 1 if inds[0] > 0 else None:-1]
            new_perm[inds[0]:inds[1] + 1] = new_perm[inds[1]:inds[0] - 1 if inds[0] > 0 else None:-1]

            perm = new_perm
            dist_arr = new_dist_arr

            res.append(perm)

        return res

    def calc_dist_array(self, points, dist):
        res = []
        for i in range(len(points) - 1):
            res.append(dist(points[i], points[i + 1]))

        return res, np.sum(res)

    def find_best_pair(self, X: np.ndarray, dist_arr, curr_loss):
        """
        It changes the range, not just 2 points
        """
        n = X.shape[0]
        best_pair = (0, 0)
        best_loss = curr_loss
        best_dist_arr = dist_arr.copy()

        for i in range(n - 1):
            # don't wanna change adjacent points -> i + 2
            for j in range(i + 2, n):
                new_loss = curr_loss
                new_dist_arr = dist_arr.copy()

                if i > 0:
                    new_loss -= dist_arr[i - 1]
                    new_dist = self.dist(X[i - 1], X[j])

                    new_loss += new_dist
                    new_dist_arr[i - 1] = new_dist

                if j + 1 < n:
                    new_loss -= dist_arr[j]
                    new_dist = self.dist(X[i], X[j + 1])

                    new_loss += new_dist
                    new_dist_arr[j] = new_dist

                if new_loss < best_loss:
                    best_pair = (i, j)
                    best_loss = new_loss
                    best_dist_arr = new_dist_arr.copy()

        return best_pair, best_loss, best_dist_arr
