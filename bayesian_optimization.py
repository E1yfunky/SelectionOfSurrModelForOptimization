import enum
import time
import warnings

import numpy as np
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import acq_max
from sklearn.base import clone
from sklearn.gaussian_process.kernels import Matern


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

        for x, f in zip(self.init_x, self.init_f):
            self.register(x, -f[0])

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

            t_start = time.time()
            if self.alpha is None:
                self._gp.fit(self._space.params, self._space.target)
                suit, score = estimate_model(self._gp, self._space.params, self._space.target)
                metric = np.nan
            else:
                suits, scores, models = [], [], []
                for nu_candidate in sorted([0.5, 1.5, 2.5, np.inf] + [0, 1, 2, 3]):
                    self.set_gp_params(kernel=Matern(nu=nu_candidate))
                    self._gp.fit(self._space.params, self._space.target)
                    models.append(clone(self._gp))
                    suit, score = estimate_model(self._gp, self._space.params, self._space.target)
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
