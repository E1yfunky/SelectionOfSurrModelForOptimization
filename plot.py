from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FOLDER = "data"
PLOTS_FOLDER = "plots"
NONE_LABEL = "NONE"

plots_path = Path(PLOTS_FOLDER)
plots_path.mkdir(parents=True, exist_ok=True)

data_list = []
for file in Path(DATA_FOLDER).iterdir():
    print(file.stem)
    data = pd.read_csv(file, index_col=0)
    data_list.append(data)
all_data = pd.concat(data_list)
all_data["alpha"] = all_data["alpha"].fillna(NONE_LABEL).astype(str)
all_data["f_min"] = all_data.groupby(["problem", "dim"], dropna=False)[["f"]].transform("min")

# %%
selection_results = all_data[all_data["alpha"] != NONE_LABEL]
initial_idx = selection_results["nu"].isnull()
selection_results = selection_results[~initial_idx]
selection_results["nu"] = selection_results["nu"].replace(np.inf, 6)
selection_results["suitability"].std()
selection_results["score"].std()

plt.plot(selection_results["score"], selection_results["suitability"], 'o')
plt.grid()
plt.xlabel("score")
plt.ylabel("suitability")
plt.savefig(plots_path / "ALL score_vs_suitability.png")
plt.show()
plt.close()

plt.hist(selection_results["nu"])
plt.grid()
plt.xlabel("score")
plt.ylabel("suitability")
plt.savefig(plots_path / f"ALL selected nu.png")
plt.show()
plt.close()

# %%
run_results = {
  "problem": [],
  "dim": [],
  "seed": [],
  "alpha": [],
  "nu_fixed": [],
  "f_best": [],
  "f_best_norm": [],
  "time": [],
}
for keys, data in all_data.groupby(["problem", "dim", "seed", "alpha", "nu_fixed"], dropna=False):
    problem, dim, seed, alpha, nu_fixed = keys
    f_min_known = data["f_min"].min()
    run_results["problem"].append(problem)
    run_results["dim"].append(dim)
    run_results["seed"].append(seed)
    run_results["alpha"].append(alpha)
    run_results["nu_fixed"].append(nu_fixed)
    run_results["f_best"].append(data["f"].min())
    run_results["f_best_norm"].append((data["f"].min() - f_min_known) / max(1, np.abs(f_min_known)))
    run_results["time"].append(data["time"].sum())
run_results = pd.DataFrame(run_results)

# %%
for (problem, dim), data in run_results.groupby(["problem", "dim"]):
    ax = data.boxplot(by='alpha', column='f_best')
    title = f"{problem} {dim}"
    ax.set_title(f"{problem} {dim}")
    ax.set_ylabel("f_best")
    plt.savefig(plots_path / f"PROBLEM {title}.png")
    plt.show()
    plt.close()

# %%
values = np.unique(run_results["f_best_norm"])
for alpha, data in run_results.groupby("alpha"):
    ratios = [sum(data["f_best_norm"] < value) for value in values]
    plt.plot(values, ratios, label=alpha)
plt.grid()
plt.legend()
plt.savefig(plots_path / "ALL performance_plots.png")
plt.show()
plt.close()
