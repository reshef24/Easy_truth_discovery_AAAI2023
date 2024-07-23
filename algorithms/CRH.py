import numpy as np
import pandas as pd
from scipy.spatial import distance

from utils.aggregations import plurality
from configurations.constants import Constants


"""
Based on the pseudo-code in: Li et. al., "Resolving Conflicts in Heterogeneous Data by Truth
Discovery and Source Reliability Estimation"
Link: https://dl.acm.org/doi/pdf/10.1145/2588555.2610509
INPUT: Continuous data
PARAMS:
        - iterations: # of repetitions of estimating the answers and weighting the workers by the estimations.
        - pairwise_distance_function: the distance function to use for calculating the distance between the answers.
"""


def CRH(iterations, data: pd.DataFrame, pairwise_distance_function):
    n, m = data.shape
    weights = np.ones(n)
    df_values = data.values
    agg_answers = np.average(df_values, axis=0, weights=weights, returned=False).reshape(1, -1)

    for iteration in range(iterations):
        d_i_t = pairwise_distance_function(df_values, agg_answers.reshape(1, -1))
        d_i_t = np.clip(d_i_t, a_min=Constants.fault_min, a_max=Constants.fault_max)
        w_i_t = -np.log(d_i_t / d_i_t.sum())
        w_i_t = w_i_t / w_i_t.sum()
        agg_answers = np.average(df_values, axis=0, weights=w_i_t.reshape(-1, ), returned=False)

    return agg_answers


"""
Based on Jing Gao et al [AAAi'14] paper.
This is the same algorithm as CRH for continuous data.

Categorical data only
"""


def PMTD(iterations, data: pd.DataFrame, positive=True):
    possible_answers = len(np.unique(data.values))
    n, m = data.shape

    weights = np.ones(n) / n
    distances = np.ones(n)

    for iter in range(iterations):
        old_weights = weights
        answers = plurality(data, possible_answers, weights, signed=False)
        for i in range(n):
            distances[i] = distance.hamming(data.iloc[i][:], answers, w=None)
        sum_dist = np.sum(distances)
        weights = np.zeros(n)
        for i in range(n):
            if distances[i] < Constants.convergence_limit:
                weights[i] = 1000000
            else:
                weights[i] = np.log(sum_dist / distances[i])
        if positive:
            weights = np.maximum(weights, 0)
        sum_weights = np.sum(weights)
        if sum_weights < Constants.convergence_limit:
            weights = np.ones(n) / n
        else:
            weights = weights / sum_weights
        if old_weights is not None and np.linalg.norm(weights - old_weights) < Constants.convergence_limit:
            break

    return answers
