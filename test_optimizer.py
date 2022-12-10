import numpy as np
import matplotlib.pyplot as plt
from optimizer import Optimizer
from mopt import mopt

dim = 2
doe = "lhs"
seed = 0
size = 15
n_iter = 10

print(mopt.problems.f1._dn) # list of problems with variable dimensionality
problem = mopt.problems.f1.rosenbrock.Problem(dim)
problem.plot()

init_sample = mopt.problems.Sample(problem, doe=doe, size=size, seed=seed, tag=f"seed={seed}", verbose=True)
print(init_sample.full_id)
plt.plot(init_sample.x[:, 0], init_sample.x[:, 1], 'o')
plt.show()

test_sample = mopt.problems.Sample(problem, doe="rnd", size=1000, seed=0, tag=f"seed={seed}", verbose=True)
print(test_sample.full_id)
plt.plot(test_sample.x[:, 0], test_sample.x[:, 1], 'o')
plt.show()

optimizer = Optimizer(problem=problem,
                      init_sample=(init_sample.x, init_sample.f),
                      test_sample=(test_sample.x, test_sample.f))

optimizer.maximize(init_points=0, n_iter=n_iter)

history_x = np.array(optimizer.history_x)
history_f_true = np.array(optimizer.history_f_true)
history_f_model = np.array(optimizer.history_f_model)
models_suitability = optimizer.models_suitability
models_score = optimizer.models_score

plt.plot(history_f_true, label="true")
plt.plot(history_f_model, label="model")
plt.ylabel("Objective values")
plt.xlabel("N iteration")
plt.legend()
plt.grid()
plt.show()

plt.plot(models_suitability, label="suitability ")
plt.plot(models_score, label="score")
plt.ylabel("Model accuracy")
plt.xlabel("N iteration")
plt.legend()
plt.grid()
plt.show()
