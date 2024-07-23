import numpy as np
import pandas as pd

from utils.aggregations import plurality
from configurations.constants import Constants

"""
 Dawid-Skene estimator.
 Implementation based on pseudocode in
 Gao, Chao, and Dengyong Zhou. "Minimax optimal convergence rates for estimating
outcome ground truth from crowdsourced labels." arXiv preprint arXiv:1310.5764 (2013).

 implementation does not include the initialization step.
 
 INPUT: Binary data only.
 PARAMS: 
        - iterations: # of repetitions of estimating the answers and weighting the workers by the estimations.
        - positive: whether to clip the weights to be positive or not.
"""


def DSTD(data: pd.DataFrame, iterations, positive=False):
    possible_answers = set(np.unique(data.values))
    if possible_answers.union({1, 0}) != possible_answers:
        raise Exception("Error: Data is not binary")
    D = np.asarray(data * 2 - 1)
    n, m = data.shape
    weights = np.ones(n)

    for iter in range(iterations):
        old_weights = weights
        soft_answers = np.matmul(weights, D)
        soft_answers_in_range = (np.exp(soft_answers) - 1) / (np.exp(soft_answers) + 1)
        estimated_competence = np.matmul(D, soft_answers_in_range) / m
        estimated_competence = np.clip((estimated_competence + 1) / 2,
                                       a_min=1 - Constants.fault_max_cat,
                                       a_max=1 - Constants.fault_min_cat)
        weights = np.log(estimated_competence / (1 - estimated_competence))
        if positive:
            weights = np.maximum(weights, 0)
        if old_weights is not None and np.linalg.norm(weights - old_weights) < Constants.convergence_limit:
            break
    answers = plurality(data, 2, weights)
    return answers
