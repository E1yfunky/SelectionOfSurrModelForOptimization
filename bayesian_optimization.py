import time
import warnings
from copy import deepcopy

import numpy as np
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import acq_max
from sklearn.gaussian_process.kernels import Matern


def collect_pairs(distances, n_neighbors, max_distance):
    pairs = set()
    for i, distances_i in enumerate(distances):
        distances_i = distances_i[i + 1:]
        neighbors = np.argsort(distances_i)
        # Select no more than specified number of points,
        # each point should not be further than the max distance
        for neighbor in neighbors[:n_neighbors]:
            if distances_i[neighbor] <= max_distance:
                pairs.add((i, neighbor + i + 1))
    return list(pairs)


def interpolate(x, f, pairs, n_points):
    x_interp, f_interp = [], []
    if n_points == 0:
        return x_interp, f_interp
    x_distances = np.linspace(0, 1, n_points + 2)[1:-1]
    for i1, i2 in pairs:
        x_interp.extend(np.linspace(x[i1], x[i2], n_points + 2)[1:-1])
        f_interp.extend(np.interp(x_distances, [0, 1], [f[i1][0], f[i2][0]]))
    return np.vstack(x_interp), np.vstack(f_interp).reshape(-1, 1)


class Sample:

    def __init__(self, init_x, init_f, n_ext):
        self.x = init_x
        self.f = init_f
        self.n_points = len(init_x)
        self.dim = len(init_x[0])
        self.n_neighbors = 2 * self.dim
        self.n_ext = n_ext

        self.distances = np.hypot.reduce(init_x - init_x[:, np.newaxis], axis=2)
        np.fill_diagonal(self.distances, np.inf)
        self.threshold = np.min(self.distances, axis=0).max()

        self.pairs = collect_pairs(distances=self.distances, n_neighbors=self.n_neighbors, max_distance=self.threshold)
        self.x_interp, self.f_interp = interpolate(x=self.x, f=self.f, pairs=self.pairs, n_points=self.n_ext)

    def update(self, new_x, new_f):
        new_x = np.atleast_2d(new_x)
        new_f = np.atleast_2d(new_f)
        distances = np.zeros((self.n_points + 1, self.n_points + 1))
        distances[:self.n_points, :self.n_points] = self.distances
        new_distances = np.hypot.reduce(new_x - self.x, axis=1)
        distances[:-1, -1] = distances[-1, :-1] = new_distances
        distances[-1, -1] = np.inf
        self.distances = distances

        self.x = np.concatenate((self.x, np.atleast_2d(new_x)))
        self.f = np.concatenate((self.f, np.atleast_2d(new_f)))
        self.n_points += 1

        self.threshold = max(self.threshold, new_distances.min())
        self.pairs = collect_pairs(self.distances, n_neighbors=self.n_neighbors, max_distance=self.threshold)
        self.x_interp, self.f_interp = interpolate(x=self.x, f=self.f, pairs=self.pairs, n_points=self.n_ext)

    @property
    def x_all(self):
        return np.vstack([self.x, self.x_interp]) if self.n_ext else self.x

    @property
    def f_all(self):
        return np.vstack([self.f, self.f_interp]) if self.n_ext else self.f


def estimate_model(model, x, f):
    f_model = model.predict(x)
    if np.isnan(f_model).all():
        # Failed to build model
        return -np.inf, -np.inf
    idx = np.triu_indices(len(x), 1)
    f_diffs = f.reshape(-1, 1) - f.flatten()
    f_model_diffs = f_model.reshape(-1, 1) - f_model
    f_comparison = np.sign(np.array(f_diffs[idx]))
    f_model_comparison = np.sign(f_model_diffs[idx])
    suitability = np.mean(f_comparison == f_model_comparison)
    score = model.score(x, f)
    return suitability, score


class Bayesian_Optimization(BayesianOptimization):

    def __init__(self, problem, init_xf, nu_fixed, alpha, n_ext, *args, **kwargs):

        self.problem = problem
        self.init_x, self.init_f = init_xf
        self.alpha = alpha
        self.sample = Sample(self.init_x, self.init_f, n_ext=n_ext)

        super().__init__(
            f=lambda **kwa: -self.problem.evaluate(self.space.params_to_array(kwa))[0][0],
            pbounds={f"x[{i}]": bound for i, bound in enumerate(zip(*problem.variables_bounds))},
            *args,
            **kwargs,
        )

        for x in self.init_x:
            # Do not register f values so that the
            # initial sample is printed to log
            self.probe(x)

        # Reinitialize internal GP regressor with default nu value
        self.set_gp_params(kernel=Matern(nu=nu_fixed))

        n_init = len(self.init_x)
        self.history_suit = [np.nan] * n_init
        self.history_score = [np.nan] * n_init
        self.history_metric = [np.nan] * n_init
        self.history_nu = [np.nan] * n_init
        self.history_x = [_.tolist() for _ in self.init_x]
        self.history_f = [_[0] for _ in self.init_f]
        self.history_f_model = [np.nan] * n_init
        self.history_time = [0] * n_init

    def suggest(self, utility_function):
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.all(self.sample.x == self._space.params)

            t_start = time.time()
            if self.alpha is None:
                self._gp.fit(self._space.params, self._space.target)
                suit, score = estimate_model(self._gp, self.sample.x_all, -self.sample.f_all)
                metric = np.nan
            else:
                suits, scores, models = [], [], []
                for nu_candidate in sorted([0.5, 1.5, 2.5, np.inf] + [1, 2, 3]):
                    self.set_gp_params(kernel=Matern(nu=nu_candidate))
                    self._gp.fit(self._space.params, self._space.target)
                    models.append(deepcopy(self._gp))
                    suit, score = estimate_model(self._gp, self.sample.x_all, -self.sample.f_all)
                    suits.append(suit)
                    scores.append(score)
                metrics = self.alpha * np.array(scores) + (1 - self.alpha) * np.array(suits)
                best_idx = np.argmax(metrics)
                suit = suits[best_idx]
                score = scores[best_idx]
                metric = metrics[best_idx]
                self._gp = models[best_idx]

            self.history_suit.append(suit)
            self.history_score.append(score)
            self.history_metric.append(metric)
            self.history_nu.append(self._gp.get_params()["kernel"].get_params()["nu"])
            self.history_time.append(time.time() - t_start)

        # Finding argmax of the acquisition function.
        suggestion_x = acq_max(ac=utility_function.utility,
                               gp=self._gp,
                               constraint=self.constraint,
                               y_max=self._space.target.max(),
                               bounds=self._space.bounds,
                               random_state=self._random_state)
        suggestion_f = float(self.problem.evaluate(suggestion_x))
        suggestion_f_model = float(-self._gp.predict([suggestion_x]))

        self.sample.update(suggestion_x, suggestion_f)
        self.history_x.append(suggestion_x)
        self.history_f.append(suggestion_f)
        self.history_f_model.append(suggestion_f_model)

        return self._space.array_to_params(suggestion_x)
