import warnings
import random

import numpy as np
import time
from bayes_opt.constraint import ConstraintModel
from bayes_opt.bayesian_optimization import Queue, BayesianOptimization

from scipy.stats import qmc
from bayes_opt.target_space import TargetSpace
from bayes_opt.event import Events, DEFAULT_EVENTS
from bayes_opt.logger import _get_default_logger
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class Bayesian_Optimization(BayesianOptimization):
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    constraint: A ConstraintModel. Note that the names of arguments of the
        constraint function and of f need to be the same.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    Methods
    -------
    probe()
        Evaluates the function on the given points.
        Can be used to guide the optimizer.

    maximize()
        Tries to find the parameters that yield the maximum value for the
        given function.

    set_bounds()
        Allows changing the lower and upper searching bounds
    """

    def __init__(self,
                 f,
                 pbounds,
                 constraint=None,
                 random_state=None,
                 verbose=2,
                 bounds_transformer=None,
                 allow_duplicate_points=True,
                 nu=2.5,
                 init_sample=None,
                 alfa=0.5,
                 isMode=False):
        self._random_state = ensure_rng(random_state)
        self._seed = random_state
        self._allow_duplicate_points = allow_duplicate_points
        self._queue = Queue()

        if init_sample is not None:
            for init_point in init_sample:
                self._queue.add(init_point)

        self._pbounds = pbounds
        self._score_res = []
        self.test_x = []
        self._res = []
        self._suit = []
        self._metric = []
        self._model_res = []
        self._time = []
        self._nu = nu
        self._alfa = alfa
        self.isMode = isMode
        if isMode:
            self._bst_nu = []

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        if constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = TargetSpace(f, pbounds, random_state=random_state,
                                      allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                constraint.fun,
                constraint.lb,
                constraint.ub,
                random_state=random_state
            )
            self._space = TargetSpace(
                f,
                pbounds,
                constraint=constraint_,
                random_state=random_state
            )
            self.is_constrained = True

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    def calculate_suit(self, gp_model):
        test_y = gp_model.predict(self.all_x)
        y_ = np.array([temp_y for temp_y in self._sample])
        idx = np.triu_indices(len(self.all_x), 1)
        y_diffs = y_.reshape(-1, 1) - y_
        test_y_diffs = test_y.reshape(-1, 1) - test_y
        y_comparison = np.sign(np.array(y_diffs[idx]))
        test_y_comparison = np.sign(test_y_diffs[idx])
        return np.mean(y_comparison == test_y_comparison)

    def find_bst_metric(self):
        nu_mas = [0, 1.25, 1.5, 2.25, 2.5, 3, np.inf]
        max_metric = best_suit = best_score = -np.inf
        temp_bst_nu = self._nu
        for nu_ in nu_mas:
            temp_gp = GaussianProcessRegressor(
                kernel=Matern(nu_),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self._random_state,
                )
            temp_gp.fit(self._space.params, self._space.target)
            suit = self.calculate_suit(temp_gp)
            score_ = self._gp.score(self.all_x, self._sample)
            temp_metric = self._alfa * score_ + (1 - self._alfa) * suit
            print(temp_metric)
            if temp_metric > max_metric:
                temp_bst_nu = nu_
                best_suit = suit
                best_score = score_
                max_metric = temp_metric
                self._gp = temp_gp
        return max_metric, temp_bst_nu, best_suit, best_score

    def probe(self, params, lazy=True, writing=False):
        """
        Evaluates the function on the given points. Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """

        if lazy:
            self._queue.add(params)
        else:
            y = self._space.probe(params)
            if writing:
                self._res.append(-y)
            self._sample.append(y)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promising point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
            if self.isMode:
                metric, temp_bst_nu, max_suit, max_score = self.find_bst_metric()
                max_score = self._gp.score(self.all_x, self._sample)
                self._bst_nu.append(temp_bst_nu)
            else:
                max_suit = self.calculate_suit(self._gp)
                max_score = self._gp.score(self.all_x, self._sample)
                metric = self._alfa * max_score + (1 - self._alfa) * max_suit

            self._metric.append(metric)
            self._suit.append(max_suit)
            self._score_res.append(max_score)
            if self.is_constrained:
                self.constraint.fit(self._space.params,
                                    self._space._constraint_values)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(ac=utility_function.utility,
                             gp=self._gp,
                             constraint=self.constraint,
                             y_max=self._space.target.max(),
                             bounds=self._space.bounds,
                             random_state=self._random_state)
        return self._space.array_to_params(suggestion)

    def hyper_cube(self, init_points):
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        d = len(self._pbounds.keys())
        data = np.empty((1, d))
        sampler = qmc.LatinHypercube(d=d)
        sample = sampler.random(n=init_points)
        for point in range(init_points):
            for col, (lower, upper) in enumerate(self._pbounds.values()):
                data.T[col] = sample[point][col] * (upper - lower) + lower
            self._queue.add(data.ravel())
            data = np.empty((1, d))

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq: {'ucb', 'ei', 'poi'}
            The acquisition method used.
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.

        kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is
                the highest.

        kappa_decay: float, optional(default=1)
            `kappa` is multiplied by this factor every iteration.

        kappa_decay_delay: int, optional(default=0)
            Number of iterations that must have passed before applying the
            decay to `kappa`.

        xi: float, optional(default=0.0)
            [unused]
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        if self._queue.empty:
            self.hyper_cube(init_points)
        self.set_gp_params(**gp_params)
        self.all_x = []
        self._sample = []

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            if not self._queue.empty:
                x_probe = next(self._queue)
                write_ = False
                self.all_x.append(x_probe)
            else:
                t_start = time.time()
                util.update_params()
                x_probe = self.suggest(util)
                self.test_x.append(x_probe)
                iteration += 1
                write_ = True
                self._time.append(time.time() - t_start)
                self.all_x.append(np.array([x_probe[key] for key in sorted(x_probe)], dtype=float))
            self.probe(x_probe, lazy=False, writing=write_)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)