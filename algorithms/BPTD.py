import copy

import numpy as np
import pandas as pd

from configurations.constants import Constants

"""
Belief Propagation Truth Discovery. Following [Karger, Oh and Shah, NeurIPS 2011].

INPUT: binary data
PARAMS: 
        - iterations: # of repetitions of estimating the answers and weighting the workers by the estimations.
        - use_random_init: whether to use random initialization of the betas or initialize them to 1. 
"""


def BPTD(iterations, data: pd.DataFrame, use_random_init=True):
    possible_answers = set(np.unique(data.values))
    if possible_answers.union({1, 0}) != possible_answers:
        raise Exception("Error: Data is not binary")

    n, m = data.shape
    betas = np.ones((n, m))
    alphas = np.zeros((n, m))
    D = np.asarray(data * 2 - 1)

    if use_random_init:
        betas = np.random.normal(1, 1, (n, m))

    for iteration in range(iterations):
        # last_iter = iter
        a_betas_old = np.average(betas, 1)
        # update alphas
        for i in range(n):
            betas_copy = copy.deepcopy(betas)
            betas_copy[i, :] = np.zeros(m)
            alphas[i, :] = np.nansum(betas_copy * D, 0) / m
        # update betas
        for j in range(m):
            alphas_copy = copy.deepcopy(alphas)
            alphas_copy[:, j] = np.zeros(n)
            betas[:, j] = np.nansum(alphas_copy * D, 1)
        a_betas = np.average(betas, 1)
        if a_betas_old is not None and np.linalg.norm(a_betas - a_betas_old) < Constants.convergence_limit:
            break

    # random tie breaking:
    final_a = np.nansum(betas * D, 0) + (np.random.rand(m) - 0.5) / 10000000
    answers = (np.sign(final_a) + 1) / 2

    return answers