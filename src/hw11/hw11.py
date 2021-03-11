import numpy as np

from typing import List


def cyclic_distance(points, dist):
    res = 0
    for i in range(len(points) - 1):
        res += dist(points[i], points[i + 1])
    res += dist(points[0], points[-1])

    return res


def l2_distance(p1, p2):
    d = p1 - p2
    return np.sqrt(d @ d)


def l1_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))


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
        res += dist(points[0], points[-1])

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


class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations

    def optimize(self, X) -> np.ndarray:
        population = self.optimize_explain(X)
        return self.fitness(X, population)[0]

    def optimize_explain(self, X) -> np.ndarray:
        n = X.shape[0]
        population = np.array([np.random.permutation(n) for _ in range(self.pop_size)]).astype(int)
        res = [population]

        for _ in range(self.iters):
            pop_inds_sorted = self.fitness(X, population)

            # take the best
            pop_inds_survived = pop_inds_sorted[:self.surv_size]

            # generate new population
            population = self.crossover(population[pop_inds_survived])

            res.append(population)

        return res

    def fitness(self, X: np.ndarray, population: np.ndarray) -> np.ndarray:
        return np.argsort([cyclic_distance(X[perm], self.dist) for perm in population])

    def crossover(self, population: List[np.ndarray]) -> np.ndarray:
        n = len(population)
        res = []

        for _ in range(self.pop_size):
            inds = np.random.choice(n, 2, replace=True)
            res.append(self.breed(*population[inds]))

        return np.array(res, dtype=int)

    def breed(self, perm1, perm2) -> np.ndarray:
        n = len(perm1)
        copy_inds = np.sort(np.random.choice(n, 2, replace=True))

        part1 = perm1[copy_inds[0]:copy_inds[1] + 1]
        part2 = [ind for ind in perm2 if ind not in part1]

        res = np.concatenate([part1, part2])
        return res
