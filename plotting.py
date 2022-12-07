import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results_by_nu(dimension, df, otn, prefix):
	fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 24))

	suit_mas = []
	target_mas = []
	score_mas = []
	for nu in np.linspace(0, 3, 13):
		suit_mas.append(df[df.nu == nu].groupby(['iteration']).agg({'suitability': 'mean'})['suitability'])
		target_mas.append(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		score_mas.append(df[df.nu == nu].groupby(['iteration']).agg({'score': 'mean'})['score'])

	ax[0].set_yscale('log')
	ax[0].boxplot(suit_mas)
	ax[0].set_xticklabels(np.linspace(0, 3, 13))
	ax[0].set_title(f'{dimension}d suitability of nu, 1:{otn}')
	ax[0].set_ylabel('Suitability')
	ax[0].set_xlabel('Nu')

	ax[1].set_yscale('log')
	ax[1].boxplot(target_mas)
	ax[1].set_xticklabels(np.linspace(0, 3, 13))
	ax[1].set_title(f'{dimension}d target of nu, 1:{otn}')
	ax[1].set_ylabel('F*')
	ax[1].set_xlabel('Nu')

	ax[2].set_yscale('log')
	ax[2].boxplot(score_mas)
	ax[2].set_xticklabels(np.linspace(0, 3, 13))
	ax[2].set_title(f'{dimension}d score of nu, 1:{otn}')
	ax[2].set_ylabel('Score')
	ax[2].set_xlabel('Nu')

	plt.savefig(f"{prefix}_results_of_nu.png")
	plt.show()


def the_best_of_mean_3d(df, dimension, prefix):
	mas = []
	f_mas = []
	nu_mas = []
	for nu in np.linspace(0, 3, 13):
		mas.extend(df[df.nu == nu].groupby(['seed']).agg({'suitability': 'mean'})['suitability'])
		f_mas.extend(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		nu_mas.extend([nu] * 10)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(nu_mas, f_mas, mas, alpha=0.5)
	ax.set_title(f'{dimension}d f of m_suitability of nu')
	ax.set_zlabel('suitability')
	ax.set_ylabel('F*')
	ax.set_xlabel('Nu')

	plt.savefig(f"{prefix}_f_of_m_suitability_of_nu.png")
	plt.show()


def the_best_of_mean_2d(df, dimension, prefix):
	mas = []
	f_mas = []
	nu_mas = []
	for nu in np.linspace(0, 3, 13):
		mas.append(df[df.nu == nu].groupby(['seed']).agg({'suitability': 'mean'})['suitability'])
		f_mas.append(df[df.nu == nu].groupby(['seed']).agg({'target': 'min'})['target'])
		nu_mas.append([nu] * 10)

	fig, axs = plt.subplots(nrows=4, ncols=4)
	fig.suptitle(f'{dimension}d f of m_suitability of nu')

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		axs[i % 4, i // 4].scatter(mas[i], f_mas[i], alpha=0.5)

	plt.savefig(f"{prefix}_f_of_m_suitability_for_nu_s.png")
	plt.show()


def suitability_history(df, dimension, prefix):
	s_mas = []
	max_t = df['iteration'].max()

	for nu in np.linspace(0, 3, 13):
		suit_mas = df[df.nu == nu].groupby(['iteration']).agg({'suitability': 'mean'})['suitability']
		max_s = -10
		temp_mas = []
		for suit in suit_mas:
			max_s = max(suit, max_s)
			temp_mas.append(max_s)
		s_mas.append(temp_mas)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	fig = plt.figure()
	ax = fig.add_subplot()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		ax.plot(np.arange(0, max_t + 1, 1), s_mas[i], color=colors[i], label=f'nu = {nu}')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Suitability')
	ax.legend()

	ax.set_title(f'{dimension}d history of suitability')
	plt.savefig(f"{prefix}_suitability_history.png")
	plt.show()


def f_history(df, dimension, prefix):
	f_mas = []
	max_t = df['iteration'].max()

	for nu in np.linspace(0, 3, 13):
		target_mas = df[df.nu == nu].groupby(['iteration']).agg({'target': 'mean'})['target']
		min_f = 100000000000000
		temp_mas = []
		for target in target_mas:
			min_f = min(target, min_f)
			temp_mas.append(min_f)
		f_mas.append(temp_mas)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	fig = plt.figure()
	ax = fig.add_subplot()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		ax.plot(np.arange(0, max_t + 1, 1), f_mas[i], color=colors[i], label=f'nu = {nu}')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('F*')
	ax.legend()

	ax.set_title(f'{dimension}d history of F')
	plt.savefig(f"{prefix}_f_history.png")
	plt.show()


def score_history(df, dimension, prefix):
	s_mas = []
	max_t = df['iteration'].max()

	for nu in np.linspace(0, 3, 13):
		suit_mas = df[df.nu == nu].groupby(['iteration']).agg({'score': 'mean'})['score']
		max_s = -10
		temp_mas = []
		for suit in suit_mas:
			max_s = max(suit, max_s)
			temp_mas.append(max_s)
		s_mas.append(temp_mas)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7bc8f6', '#006400', '#e6e6fa']

	fig = plt.figure()
	ax = fig.add_subplot()

	for i, nu in enumerate(np.linspace(0, 3, 13)):
		ax.plot(np.arange(0, max_t + 1, 1), s_mas[i], color=colors[i], label=f'nu = {nu}')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Score')
	ax.legend()

	ax.set_title(f'{dimension}d history of score')
	plt.savefig(f"{prefix}_score_history.png")
	plt.show()


def main():
	print("get started")
	dimension = 2
	otn = 3
	func = 'Rosenbrock'
	prefix = f"{func}_{d}d"
	df = pd.read_csv(f"{prefix}_test_data.csv",  delimiter=';')

	plot_results_by_nu(dimension, df, otn, prefix)
	f_history(df, dimension, prefix)
	the_best_of_mean_2d(df, dimension, prefix)
	the_best_of_mean_3d(df, dimension, prefix)
	suitability_history(df, dimension, prefix)


if __name__ == '__main__':
	main()