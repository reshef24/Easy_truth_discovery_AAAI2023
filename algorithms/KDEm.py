import numpy as np
import pandas as pd

from configurations.constants import Constants
from utils.aggregations import aggregate

"""
From Truth Discovery to Trustworthy Opinion Discovery:
An Uncertainty-Aware Quantitative Modeling Approach

Mengting Wan et. al

http://hanj.cs.illinois.edu/pdf/kdd16_mwan.pdf
"""

MAX_ITERATIONS = 100


def KDEm(data: pd.DataFrame,
         distance_function,
         iterations,
         mode,
         kernel,
         cluster,
         params,
         voting_rule=None,
         gt=None):
    n, m = data.shape
    possible_answers = set(np.unique(data.values))
    w_0 = np.ones(n) / n

    df_values = data.values
    np_kernel = kernel(df_values)

    max_iterations = iterations
    distance_function = distance_function
    norm = Norms[distance_function]

    loss = float("inf")
    new_loss, c_j, w_j = KDEm_iteration(np_kernel, w_0, norm)
    iter = 0
    while abs(loss - new_loss) > Constants.convergence_limit and iter < max_iterations:
        loss = new_loss
        new_loss, c_j, w_j = KDEm_iteration(np_kernel, w_0, norm)
        iter += 1

    # change to `variant'?
    if mode == "continuous":
        agg_answers, w_j = cluster_aggregation(m, df_values, w_j, cluster)

    # not in the paper:
    elif mode == "ranking":
        agg_answers = aggregate(data, gt, w_j, voting_rule, mode, possible_answers, params)

    # "best dictator per question"
    elif mode == "categorical":
        representative = np.argmax((np_kernel * w_j.reshape(-1, 1)), axis=0)
        w_j[~np.isin(np.arange(len(w_j)), representative)] = 0
        agg_answers = df_values[representative, list(range(m))]
    else:
        raise Exception("Not Implemented")

    return agg_answers


def cluster_aggregation(m, df_values, w_j, cluster):
    t_i = np.zeros(m)
    new_t = np.ones(m)
    h_i = np.median(abs(df_values - np.median(df_values, axis=0)), axis=0) + 1e-10 * np.std(df_values, axis=0)

    iteration = 0
    while abs(new_t - t_i).any() > Constants.convergence_limit and iteration < MAX_ITERATIONS:
        t_i = new_t
        cluster_kernel = np.clip(cluster((df_values - t_i) / h_i), a_min=Constants.fault_min, a_max=Constants.fault_max)
        new_t = (w_j.reshape(-1, 1) * cluster_kernel * df_values).sum(axis=0) / cluster_kernel.sum(axis=0)
        iteration += 1
    weights = (w_j.reshape(-1, 1) * cluster_kernel).sum(axis=0) / cluster_kernel.sum(axis=0)
    return t_i, weights / weights.sum()


def gaussian_kernel(x):
    return np.exp(-x ** 2) / np.sqrt(2 * np.pi)


def laplace_kernel(x):
    return np.exp(-abs(x))


def triweight_kernel(x):
    return 35 / 32 * (1 - x ** 2) ** 3 * (abs(x) <= 1)


def biweight_kernel(x):
    return 15 / 16 * (1 - x ** 2) ** 2 * (abs(x) <= 1)


def epanechnikov_kernel(x):
    return 3 / 4 * (1 - x ** 2) * (abs(x) <= 1)


def uniform_kernel(x):
    return (abs(x) <= 1) / 2


def KDEm_iteration(np_kernel, w, norm):
    f_i = (np_kernel * w.reshape(-1, 1)).sum(axis=0)
    distance_from_kernel = norm(np_kernel.T - f_i.reshape(-1, 1))
    c_j = - np.log(distance_from_kernel.sum(axis=0) / distance_from_kernel.sum())
    w_j = c_j / c_j.sum()
    loss = (c_j * distance_from_kernel).sum()
    return loss, c_j, w_j


Kernels = {
    "gaussian": np.vectorize(gaussian_kernel),
    "laplace": np.vectorize(laplace_kernel),
    "triweight": np.vectorize(triweight_kernel),
    "biweight": np.vectorize(biweight_kernel),
    "epanechnikov": np.vectorize(epanechnikov_kernel),
    "uniform": np.vectorize(uniform_kernel)
}
Norms = {
    "square_euclidean": lambda x: x ** 2,
}

Cluster = {
    "uniform" :  Kernels["uniform"],
    "epanechnikov": Kernels["uniform"],
    "biweight": Kernels["epanechnikov"],
    "triweight": Kernels["biweight"],
    "gaussian": Kernels["gaussian"]
}
