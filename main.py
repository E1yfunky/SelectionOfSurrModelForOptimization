import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_optimization import Bayesian_Optimization
from mopt import problems

DATA_FOLDER = "data"
LOG_FOLDER = "logs"

Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
log_path = Path(LOG_FOLDER) / f"main {datetime.datetime.now():%Y.%m.%d %H.%M}.log"
formatter = logging.Formatter('[%(asctime)s] <%(levelname)s> %(message)s')
log_to_stdout = logging.StreamHandler()
log_to_stdout.setFormatter(formatter)
log_to_file = logging.FileHandler(str(log_path), mode='w')
log_to_file.setFormatter(formatter)

LOG = logging.getLogger()
LOG.addHandler(log_to_file)
LOG.addHandler(log_to_stdout)
LOG.setLevel(logging.DEBUG)


def bayes_optim(problem, n_init, n_iter, n_ext, n_random_runs, nu_fixed, alpha):
    df_dct = {
        # Problem definition
        'problem': [],
        'dim': [],
        'n_init': [],
        'n_iter': [],
        'n_ext': [],
        'nu_fixed': [],
        'alpha': [],
        # Current run info
        'seed': [],
        'iteration': [],
        'suitability': [],
        'score': [],
        'metric': [],
        'nu': [],
        'x': [],
        'f': [],
        'f_model': [],
        'time': [],
    }
    for i in range(n_random_runs):
        LOG.info(f"[{i+1}/{n_random_runs}] {problem.NAME}")
        init_sample = problems.Sample(problem, doe="lhs", size=n_init, seed=i, tag=f"seed={i}", verbose=True)
        optimizer = Bayesian_Optimization(
            problem=problem,
            init_xf=(init_sample.x, init_sample.f),
            nu_fixed=nu_fixed,
            alpha=alpha,
            n_ext=n_ext,
            random_state=i,
        )
        optimizer.maximize(init_points=0, n_iter=n_iter)

        # Problem definition
        df_dct['problem'].extend([problem.NAME] * (n_init + n_iter))
        df_dct['dim'].extend([problem.size_x] * (n_init + n_iter))
        df_dct['n_init'].extend([n_init] * (n_init + n_iter))
        df_dct['n_iter'].extend([n_iter] * (n_init + n_iter))
        df_dct['n_ext'].extend([n_ext] * (n_init + n_iter))
        df_dct['nu_fixed'].extend([nu_fixed] * (n_init + n_iter))
        df_dct['alpha'].extend([alpha] * (n_init + n_iter))
        # Current run info
        df_dct['seed'].extend([i] * (n_init + n_iter))
        df_dct['iteration'].extend(list(range(1, n_init + n_iter + 1)))
        df_dct['suitability'].extend(optimizer.history_suit)
        df_dct['score'].extend(optimizer.history_score)
        df_dct['metric'].extend(optimizer.history_metric)
        df_dct['nu'].extend(optimizer.history_nu)
        df_dct['x'].extend(optimizer.history_x)
        df_dct['f'].extend(optimizer.history_f)
        df_dct['f_model'].extend(optimizer.history_f_model)
        df_dct['time'].extend(optimizer.history_time)

        best_idx = np.argmin(optimizer.history_f)
        LOG.info(
            f"[{i+1}/{n_random_runs}] {problem.NAME} "
            f"x*={optimizer.history_x[best_idx]}; "
            f"f(x*)={optimizer.history_f[best_idx]}"
        )
    return pd.DataFrame(df_dct)


def main():
    problems_list = [
        problems.f1.ackley,
        # problems.f1.bohachevsky,
        # problems.f1.griewank,
        problems.f1.levy3,
        # problems.f1.michalewicz,
        # problems.f1.perm,
        problems.f1.rastrigin,
        problems.f1.rosenbrock,
        # problems.f1.salomon,
        problems.f1.styblinskitang,
        # problems.f1.trid,
        # problems.f1.weierstrass,
    ]

    nu_fixed = 2.5
    n_ext = 0

    data_folder = Path(DATA_FOLDER)
    data_folder.mkdir(parents=True, exist_ok=True)

    for dim, n_init, n_runs in [(2, 16, 30),
                                (4, 32, 20),
                                (8, 96, 10)]:
        for problem_class in problems_list:
            for alpha in [None, 0.01, 0.5, 0.99]:
                n_iter = 2 * n_init
                problem = problem_class.Problem(dim)

                tag = f"{problem.NAME} dim={dim} n_init={n_init} n_iter={n_iter} alpha={alpha} nu_fixed={nu_fixed} n_runs={n_runs} n_ext={n_ext}"
                data_file = data_folder / f"{tag}.csv"
                LOG.info(f"{tag}")
                if data_file.exists():
                    continue

                data = bayes_optim(problem=problem,
                                   n_init=n_init,
                                   n_iter=n_iter,
                                   n_ext=n_ext,
                                   n_random_runs=n_runs,
                                   nu_fixed=nu_fixed,
                                   alpha=alpha)
                data.to_csv(data_file)


if __name__ == '__main__':
    main()
