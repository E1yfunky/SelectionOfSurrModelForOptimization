import numpy as np
from bayes_opt import BayesianOptimization

class Optimizer(BayesianOptimization):

  def __init__(self, init_sample, test_sample, problem, *args, **kwargs):
    self.problem = problem
    self.init_x, self.init_f = init_sample
    self.test_x, self.test_f = test_sample

    test_f_diffs = self.test_f - self.test_f.T
    self.test_f_diffs_idx = np.triu_indices(test_f_diffs.shape[0], 1)
    self.test_f_comparison = np.sign(test_f_diffs[self.test_f_diffs_idx])

    self.history_x = []
    self.history_f_true = []
    self.history_f_model = []

    self.models_score = []
    self.models_suitability = []
    pbounds = {f"x{i+1}": bound for i, bound in enumerate(zip(*problem.variables_bounds))}
    super().__init__(f=self.objective, pbounds=pbounds, *args, **kwargs)

    for x, f in zip(self.init_x, self.init_f):
      self.register(x, -f[0])

  def objective(self, **kwargs):
    x = self.space.params_to_array(kwargs)
    return -self.problem.evaluate(x)[0][0]

  def suggest(self, utility_function):
    next_point = super().suggest(utility_function)  # rebuild model
    self.models_score.append(self._gp.score(self.test_x, -self.test_f))

    test_f_gp = -self._gp.predict(self.test_x)
    test_f_gp_diffs = test_f_gp.reshape(-1, 1) - test_f_gp
    test_f_gp_comparison = np.sign(test_f_gp_diffs[self.test_f_diffs_idx])
    self.models_suitability.append(np.mean(test_f_gp_comparison == self.test_f_comparison))

    next_x = self.space.params_to_array(next_point)
    self.history_x.append(next_x)
    self.history_f_true.append(self.problem.evaluate(next_x)[0])
    self.history_f_model.append(-self._gp.predict([next_x]))
    return next_point
