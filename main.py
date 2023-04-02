import openpyxl
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from celluloid import Camera
import mopt
from functools import lru_cache
from bayesian_optimization import Bayesian_Optimization


@lru_cache(maxsize=None)
def black_box_func(**X):
	X = np.array([X[key] for key in sorted(X)], dtype=float)

	return -problem.define_objectives(X)[0]


def bayes_optim(d, nu_mas, init_points, n_iter, x_range, n, alfa, isMode):
	result_data = []
	df_dct = {
			  'dimension': [d] * n_iter * n * len(nu_mas),
			  'nu': [],
			  'iteration': [i for i in range(n_iter)] * len(nu_mas) * n,
			  'init_points': [init_points] * n_iter * len(nu_mas) * n,
			  'iters:points': [n_iter / init_points] * n_iter * len(nu_mas) * n,
			  'X': [],
			  'target': [],
			  'score': [],
			  'suitability': [],
			  'metric': [],
			  'seed': [],
			  'time': []}
	for nu in nu_mas:
		for i in range(n):
			seed = i
			if not isMode:
				df_dct['nu'].extend([nu] * n_iter)

			init_sample = mopt.problems.Sample(problem, doe="lhs", size=init_points, seed=seed, tag=f"seed={seed}",
											   verbose=True)
			optimizer = Bayesian_Optimization(f=black_box_func,
											 pbounds={f"x[{_}]": x_range for _ in range(d)},
											 verbose=2,
											 random_state=seed,
											 nu=nu,
											 init_sample=init_sample.x,
											 alfa=alfa,
											 isMode=isMode)

			optimizer.maximize(init_points=init_points, n_iter=n_iter)
			if isMode:
				df_dct['nu'].extend(optimizer._bst_nu)
			df_dct['X'].extend(optimizer.test_x)
			df_dct['target'].extend(optimizer._res)
			df_dct['score'].extend(optimizer._score_res)
			df_dct['metric'].extend(optimizer._metric)
			df_dct['suitability'].extend(optimizer._suit)
			df_dct['time'].extend(optimizer._time)
			df_dct['seed'].extend([seed] * n_iter)
			if len(result_data) > 0 and len(result_data[-1]) % n > 0:
				result_data[-1].append(-optimizer.max["target"])
			else:
				result_data.append([-optimizer.max["target"]])
			print(black_box_func.cache_info())
			print("{}/{} Best result: {}; f(x) = {}.".format(i + 1, n, optimizer.max["params"], optimizer.max["target"]))

	return nu_mas, np.array(result_data), df_dct


def main():
	black_box_func.cache_clear()
	global problem

	seed = 2
	random.seed(seed)
	functions = {'himmelblau': mopt.problems.f1.himmelblau, "ackley": mopt.problems.f1.ackley, 'levy': mopt.problems.f1.levy3}

	x_range = [-3, 3]
	min_nu = 0
	max_nu = 3
	otn = 3
	# nu_mas = np.linspace(min_nu, max_nu, 13)
	nu_mas = [2.5]
	d_dct = {2: 12, 4: 80, 8: 160}  #2: 12, 4: 80, 8: 160

	for d, points in d_dct.items():
		for func, problem_object in functions.items():
			n_inter = otn * points
			problem = problem_object.Problem(d)
			X, y_s, temp_df_dct = bayes_optim(d, nu_mas, points, n_inter, x_range, 20, 0.5, False)
			black_box_func.cache_clear()
			df_marks = pd.DataFrame(temp_df_dct)

			df_marks.to_csv(f'{func}_{d}d_test_data.csv', header=True, sep=';')
			print('DataFrame is written successfully to csv.')

			X, y_s, temp_df_dct = bayes_optim(d, nu_mas, points, n_inter, x_range, 20, 0.5, True)
			black_box_func.cache_clear()
			df_marks = pd.DataFrame(temp_df_dct)

			df_marks.to_csv(f'{func}_{d}d_test_data_s.csv', header=True, sep=';')
			print('DataFrame is written successfully to csv.')


if __name__ == '__main__':
	main()