import copy

import numpy as np
import pandas as pd

from utils.aggregations import plurality
from measures.pairwise_distance_functions import compute_pairwise_distances_discrete_per_question
from utils.helpers import compute_weight_from_fault, estimate_fault_from_proxy
from configurations.constants import Constants

"""
 Similar to IPTD, but with n*m latent variables.

INPUT: categorical data [Current implementation]
"""
# Not mentioned in the paper.


def EIPTD(data: pd.DataFrame, iterations, weight_transform, mu_hat, Lambda, UR):
    possible_answers = set(np.unique(data.values))
    mode = "categorical"
    n, m = data.shape

    all_weights = np.ones((n, m))
    all_proxy_score = np.ones((n, m))
    dist_matrix = compute_pairwise_distances_discrete_per_question(data) / (m - 1)
    for iter in range(iterations):
        old_weights = copy.deepcopy(all_weights)
        sum_old_w = np.nansum(old_weights, axis=0)
        for j in range(m):
            for i in range(n):
                sum_wj_except_i = (sum_old_w[j] - old_weights[i, j])
                if sum_wj_except_i == 0:
                    w_norm = 0
                else:
                    w_norm = old_weights[:, j] / sum_wj_except_i
                # override and abort if weights are mostly negative or nan
                sum_w_norm = np.sum(w_norm)
                if sum_w_norm <= 0 or np.isnan(sum_w_norm):
                    w_norm = np.ones(n) / (n - 1)
                w_norm[i] = 0
                dot = np.dot(dist_matrix[i, :, j], w_norm)
                all_proxy_score[i, j] = dot

            max_dist = max(all_proxy_score[:, j])
            all_weights[:, j] = max_dist - all_proxy_score[:, j]
        if old_weights is not None and np.linalg.norm(all_weights - old_weights) < Constants.convergence_limit:
            break
    answers = np.ones(m)
    for j in range(m):
        estimated_fault = estimate_fault_from_proxy(all_proxy_score[:, j], mode, mu_hat, Lambda, UR, possible_answers)
        final_weights_j = compute_weight_from_fault(estimated_fault, mode, positive=False,
                                                    normalize=True,
                                                    weight_transform=weight_transform,
                                                    possible_answers=possible_answers)
        answers[j] = plurality(data.iloc[:, j], possible_answers, final_weights_j)

    return answers
