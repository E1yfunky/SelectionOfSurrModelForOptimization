from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FOLDER = "data1"
PLOTS_FOLDER = "plots"
NONE_LABEL = "NONE"

plots_path = Path(PLOTS_FOLDER)
plots_path.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['font.size'] = '22'
plt.figure(figsize=(10, 8))

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

for n_ext, data in selection_results.groupby("n_ext"):
    plt.plot(data["score"], data["suitability"], 'o', alpha=0.13)
    plt.grid()
    plt.xlabel("Sc", style='italic')
    plt.ylabel("Su", style='italic')
    plt.savefig(plots_path / f"ALL score_vs_suitability n_ext={n_ext}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist(data["nu"])
    plt.grid()
    plt.xlabel('$\\nu$', style='italic')
    plt.ylabel('N', style='italic')
    plt.savefig(plots_path / f"ALL selected_nu n_ext={n_ext}.png", bbox_inches='tight')
    plt.show()
    plt.close()
# %%
run_results = {
  "problem": [],
  "dim": [],
  "seed": [],
  "alpha": [],
  "nu_fixed": [],
  "n_ext": [],
  "f_best": [],
  "f_best_norm": [],
  "time": [],
}
for keys, data in all_data.groupby(["problem", "dim", "seed", "alpha", "nu_fixed", "n_ext"], dropna=False):
    problem, dim, seed, alpha, nu_fixed, n_ext = keys
    f_min_known = data["f_min"].min()
    run_results["problem"].append(problem)
    run_results["dim"].append(dim)
    run_results["seed"].append(seed)
    run_results["alpha"].append(alpha)
    run_results["nu_fixed"].append(nu_fixed)
    run_results["n_ext"].append(n_ext)
    run_results["f_best"].append(data["f"].min())
    run_results["f_best_norm"].append((data["f"].min() - f_min_known) / max(1, np.abs(f_min_known)))
    run_results["time"].append(data["time"].sum())
run_results = pd.DataFrame(run_results)

# %%
for (problem, dim, n_ext), data in run_results.groupby(["problem", "dim", "n_ext"]):
    ax = data.boxplot(by='alpha', column='f_best')
    title = f"{problem} D{dim} n_ext={n_ext}"
    ax.set_title('')
    ax.set_ylabel('$\widetilde{f}^*$')
    ax.set_xlabel('$\\alpha$')
    plt.savefig(plots_path / f"PROBLEM {title}.png", bbox_inches='tight')
    plt.show()
    plt.close()

# %%
for n_ext, data_n_ext in run_results.groupby("n_ext"):
  values = np.unique(data_n_ext["f_best_norm"])
  plt.figure(figsize=(7, 7))
  for alpha, data_alpha in data_n_ext.groupby("alpha"):
      ratios = [sum(data_alpha["f_best_norm"] < value) for value in values]
      plt.plot(values, ratios, label=f"$\\alpha$ = {alpha}")
  plt.grid()
  plt.legend()
  plt.xlabel("$||\widetilde{f}^*||$")
  plt.ylabel("N", style='italic')
  plt.savefig(plots_path / f"ALL performance_plots n_ext={n_ext}.png")
  plt.show()
  plt.close()
