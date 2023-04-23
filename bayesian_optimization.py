import time
import warnings

import numpy as np
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import acq_max
from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern


def distance(point1, point2, target1, target2):
    square = np.square(point1 - point2)
    square = np.append(square, (target1 - target2) ** 2)
    return np.sqrt(np.sum(square))


def point_dist(i, dist_matrix, points, target_values):
    for j in range(i + 1, points.shape[0]):
        dist = distance(points[i], points[j], target_values[i], target_values[j])
        dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


def add_point(dist_matrix, n, points, target_values, nearest_points, interpolated_points, interpolated_values):
    num_points = points.shape[0]
    m = num_points - dist_matrix.shape[0]
    new_mas = [[0] * num_points] * m

    dist_matrix = np.hstack((dist_matrix, [[0] * m] * dist_matrix.shape[0]))
    dist_matrix = np.append(dist_matrix, new_mas, axis=0)

    for i in range(num_points - m, num_points):
        dist_matrix = point_dist(0, dist_matrix, points, target_values)

    nearest_points.extend([-1] * m)

    for i in range(num_points - m, num_points):
        nearest_points = find_nearest(i, dist_matrix, nearest_points)

        # Шаг 3: Интерполировать между парами точек и сохранить результат
        interpolation_vectors, interpolation_values = interpol_for_point(i, n, nearest_points, points, target_values)

        interpolated_points = np.append(interpolated_points, interpolation_vectors, axis=0)
        interpolated_values = np.append(interpolated_values, interpolation_values)

    return dist_matrix, nearest_points, interpolated_points, interpolated_values


def find_nearest(i, dist_matrix, nearest_points):
    nearest_indices = np.argsort(dist_matrix[i, :])
    nearest = nearest_indices[1] if nearest_indices[0] == i else nearest_indices[0]
    if nearest_points[nearest] == i:
        nearest = nearest_indices[2]

    nearest_points[i] = nearest
    return nearest_points


def euclidean_distance(point_a, point_b):
    """
    Вычисляет евклидово расстояние между двумя точками point_a и point_b.

    :param point_a: np.array, первая точка
    :param point_b: np.array, вторая точка
    :return: float, евклидово расстояние между точками
    """
    return np.sqrt(np.sum((point_a - point_b) ** 2))


def linear_interpolation(point_a, point_b, target_a, target_b, new_points):
    """
    Выполняет линейную интерполяцию между двумя многомерными точками point_a и point_b,
    используя расстояния между точками для вычисления параметра t.

    :param point_a: np.array, начальная точка
    :param point_b: np.array, конечная точка
    :param new_point: np.array, новая точка для интерполяции
    :return: np.array, интерполированная точка
    """
    interpolation_values = np.zeros(new_points.shape[0])
    if len(point_a) != len(point_b):
        raise ValueError("Размерности точек должны совпадать")

    distance_ab = euclidean_distance(point_a, point_b)
    if distance_ab == 0:
        raise ValueError("Начальная и конечная точки не должны совпадать")

    for i, new_point in enumerate(new_points):
        distance_an = euclidean_distance(point_a, new_point)

        t = distance_an / distance_ab
        if t < 0 or t > 1:
            raise ValueError("Новая точка должна находиться между начальной и конечной точками")

        interpolation_values[i] = (1 - t) * target_a + t * target_b
    return interpolation_values


def interpol_for_point(i, n, nearest_points, points, target_values):
    interpolation_vectors = np.linspace(points[i], points[nearest_points[i]], n + 2)[1:-1]
    interpolation_values = linear_interpolation(points[i], points[nearest_points[i]], target_values[i], target_values[nearest_points[i]], interpolation_vectors)

    return interpolation_vectors, interpolation_values


def init_function(points, target_values, n):
    num_points = points.shape[0]
    dist_matrix = np.zeros((num_points, num_points))

    # Шаг 1: Вычислить расстояния между точками и сохранить их в матрице расстояний
    for i in range(num_points):
        dist_matrix = point_dist(i, dist_matrix, points, target_values)


    nearest_points = [-1] * num_points
    # interpolated_points = interpolated_values = np.array([])

    nearest_points = find_nearest(0, dist_matrix, nearest_points)
    interpolated_points, interpolated_values = interpol_for_point(0, n, nearest_points, points, target_values)

    # Шаг 2: Найти ближайшую точку для каждой точки (или вторую по близости, если это необходимо)
    for i in range(1, num_points):
        nearest_points = find_nearest(i, dist_matrix, nearest_points)

        # Шаг 3: Интерполировать между парами точек и сохранить результат
        interpolation_vectors, interpolation_values = interpol_for_point(i, n, nearest_points, points, target_values)

        interpolated_points = np.append(interpolated_points, interpolation_vectors, axis=0)
        interpolated_values = np.append(interpolated_values, interpolation_values)
    return dist_matrix, nearest_points, interpolated_points, interpolated_values


def estimate_model(model, x, f):
    f_model = model.predict(x)
    if np.isnan(f_model).all():
        # Failed to build model
        return -np.inf, -np.inf
    idx = np.triu_indices(len(x), 1)
    f_diffs = f.reshape(-1, 1) - f
    f_model_diffs = f_model.reshape(-1, 1) - f_model
    f_comparison = np.sign(np.array(f_diffs[idx]))
    f_model_comparison = np.sign(f_model_diffs[idx])
    suitability = np.mean(f_comparison == f_model_comparison)
    score = model.score(x, f)
    return suitability, score


class Bayesian_Optimization(BayesianOptimization):

    def __init__(self, problem, init_xf, nu_fixed, alpha, *args, **kwargs):

        self.problem = problem
        self.init_x, self.init_f = init_xf
        self.alpha = alpha

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
        self.dist_matrix, self.nearest_points, self.interpolated_points, self.interpolated_values = init_function(self.init_x, self.init_f.reshape(self.init_f.shape[0],), 5 - 2)

    def suggest(self, utility_function):
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.dist_matrix.shape[0] != self._space.params.shape[0]:
                self.dist_matrix, self.nearest_points, self.interpolated_points, self.interpolated_values = add_point(self.dist_matrix, 3, self._space.params, self._space.target, self.nearest_points, self.interpolated_points, self.interpolated_values)

            t_start = time.time()
            if self.alpha is None:
                self._gp.fit(self._space.params, self._space.target)
                suit, score = estimate_model(self._gp, self._space.params, self._space.target)
                metric = np.nan
            else:
                suits, scores, models = [], [], []
                for nu_candidate in sorted([0.5, 1.5, 2.5, np.inf] + [1, 2, 3]):
                    self.set_gp_params(kernel=Matern(nu=nu_candidate))
                    self._gp.fit(self._space.params, self._space.target)
                    models.append(clone(self._gp))
                    suit, score = estimate_model(self._gp, self.interpolated_points, self.interpolated_values)
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
        suggestion = acq_max(ac=utility_function.utility,
                             gp=self._gp,
                             constraint=self.constraint,
                             y_max=self._space.target.max(),
                             bounds=self._space.bounds,
                             random_state=self._random_state)

        self.history_x.append(suggestion)
        self.history_f.append(float(self.problem.evaluate(suggestion)))
        self.history_f_model.append(float(-self._gp.predict([suggestion])))
        return self._space.array_to_params(suggestion)
