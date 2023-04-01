import openpyxl
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from celluloid import Camera
import mopt
from functools import lru_cache
from bayesian_optimization import Bayesian_Optimization


def get_one_rosenbrock(X, n):
	temp_y = 0
	for i in range(0, n - 1):
		temp_y += (1 - X[i]) ** 2 + 100 * ((X[i + 1] - X[i] ** 2) ** 2)

	return temp_y


def get_rosenbrock_from_data(m, n, X):
	"""
	Возвращает m результатов функции Розенброка для заданных точек X размерностей n
	:param m: число точек (целое)
	:param n: размерность вектора параметров (целое)
	:param X: двумерный массив точек
	:return: одномерный массив значений функции
	"""
	y = np.empty((m,))
	for j in range(0, m):
		y[j] = (np.array([- get_one_rosenbrock(X[j], n)]))

	return y


def make_points_animation(optimizer, iters, x_range):
	fig = plt.figure()
	ax = fig.add_subplot()
	camera = Camera(fig)

	test_data = optimizer.test_x
	test_x = np.array([i[0] for i in test_data[:-iters]], dtype=float)
	test_points = optimizer.test_y[:-iters]

	x_ros = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
	z_ros = []
	for j in range(len(x_ros)):
		z_ros.append(
			list(get_rosenbrok_from_data(len(x_ros), 2, [[x_ros[j]] for i in range(len(x_ros))])))

	for i in range(len(test_points)):
		temp_x = test_x[:i + 1]
		temp_z = test_points[:i + 1]
		ax.scatter(temp_x, temp_z, color='green')
		ax.plot(x_ros, - np.array(z_ros), cmap='inferno')
		camera.snap()

	animation = camera.animate()
	animation.save('my_2animation.gif')


def make_3d_points_animation(optimizer, iters, x_range):
	fig, ax = plt.subplots()

	camera = Camera(fig)

	test_data = optimizer.test_x
	test_data = np.array([np.array([key for key in i], dtype=float) for i in test_data[:-iters]])
	test_x = np.array([i[0] for i in test_data], dtype=float)
	test_y = np.array([i[1] for i in test_data], dtype=float)

	x_ros = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
	y_ros = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
	x_ros, y_ros = np.meshgrid(x_ros, y_ros)
	z_ros = []
	for j in range(len(x_ros)):
		z_ros.append(list(
			get_rosenbrock_from_data(len(x_ros[j]), 2, [[x_ros[j][i], y_ros[j][i]] for i in range(len(x_ros[j]))])))

	for i in range(len(test_x)):
		temp_x = test_x[:i + 1]
		temp_y = test_y[:i + 1]

		lev = [0, 0.1, 0.14, 0.17, 0.24, 0.3, 0.7, 1, 6, 10, 40, 100, 900, 2500]
		color_region = np.zeros((14, 3))
		color_region[:, 1:] = 0.8
		color_region[:, 0] = np.linspace(0, 1, 14)
		ax.contourf(x_ros, y_ros, - np.array(z_ros), levels=lev, colors=color_region)
		ax.scatter(temp_x, temp_y, color='red')
		camera.snap()

	animation = camera.animate()
	animation.save('my_animation_centered.gif')


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
	func = "ackley"

	x_range = [-3, 3]
	min_nu = 0
	max_nu = 3
	otn = 3
	# nu_mas = np.linspace(min_nu, max_nu, 13)
	nu_mas = [2.5]
	d_dct = {2: 12, 4: 80, 8: 160}  #2: 12, 4: 80, 8: 160

	for d, points in d_dct.items():
		n_inter = otn * points
		problem = mopt.problems.f1.ackley.Problem(d)
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