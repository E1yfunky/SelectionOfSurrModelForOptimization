from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FOLDER = "data"

path = Path(DATA_FOLDER)

data_list = []
for file in path.iterdir():
    print(file.stem)
    data = pd.read_csv(file, index_col=0)
    data_list.append(data)
all_data = pd.concat(data_list)
all_data["alpha"] = all_data["alpha"].fillna('NONE').astype(str)


# %%
run_results = {
  "problem": [],
  "dim": [],
  "seed": [],
  "alpha": [],
  "nu_fixed": [],
  "f_best": [],
  "time": [],
}
for keys, data in all_data.groupby(["problem", "dim", "seed", "alpha", "nu_fixed"], dropna=False):
    problem, dim, seed, alpha, nu_fixed = keys
    run_results["problem"].append(problem)
    run_results["dim"].append(dim)
    run_results["seed"].append(seed)
    run_results["alpha"].append(alpha)
    run_results["nu_fixed"].append(nu_fixed)
    run_results["f_best"].append(data["f"].min())
    run_results["time"].append(data["time"].sum())
run_results = pd.DataFrame(run_results)

for problem, data in run_results.groupby("problem"):
    ax = data.boxplot(by='alpha', column='f_best')

# %%
values = np.unique(run_results["f_best"])
for alpha, data in run_results.groupby("alpha"):
    ratios = [sum(data["f_best"] < value) for value in values]
    plt.plot(values, ratios, label=alpha)
plt.grid()
plt.legend()
plt.show()
