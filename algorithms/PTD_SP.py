import numpy as np
import pandas as pd

from configurations.constants import Constants
from utils.aggregations import aggregate
from utils.helpers import compute_weight_from_fault, estimate_fault_from_proxy, compute_weighted_average_pairwise

"""
[Iterative] Proximity-based Truth Discovery. 
variants:
* pairwise_proximity (PP): competence_j = avg(p(i,j)); weight_j = competence_j
* pairwise_distance (PD):  fault_j = avg(d(i,j));  weight_j = max(fault) - fault_j 
* AWG     (only in continuous mode)
* IER     (only in categorical mode)
* IER_BIN  (only in binary/ranking mode)
* IER_SP  Surprisingly popular (only in binary mode)

INPUT: Any distance-based data 
"""
# Not published in the paper.

def SP_binary(data: pd.DataFrame, meta):
    n, m = data.shape
    answers = np.zeros(m)
    tie_break = np.random.rand(m) / 1000
    for item_j in range(m):
        sum_meta_j = np.nanmean(meta.iloc[:, item_j])
        sum_answer_j_pos = np.nansum(data.iloc[:, item_j])
        sum_answer_j_neg = np.nansum(1 - data.iloc[:, item_j])
        if sum_answer_j_pos / (sum_answer_j_pos + sum_answer_j_neg) + tie_break[item_j] > sum_meta_j:
            answers[item_j] = 1

    return answers


def IPTD_SP(data: pd.DataFrame,
            gt,
            voting_rule,
            meta,
            iterations,
            mode,
            SP_variant,
            PTD_variant,
            mu_hat,
            Lambda,
            UR,
            params,
            positive=True,
            weight_transform="simple",
            SP_aggregate=False,
            normalize=True):
    # compute (weighted) average distance to other workers, weighing
    if mode == "rankings":
        possible_answers = 2
    else:
        possible_answers = set(np.unique(data.values))

    # pairwise_relation = data_params.get('pairwise_relation','distance')  #  'distance' or 'proximity'
    n, m = data.shape
    A = data.values
    M = meta.values
    nan_map = 1.0 - np.isnan(A)
    nan_map[nan_map == 0] = np.nan

    # [i1,i2,j] = True if i1,i2 agree on itm j
    per_item_diff_tensor = A[:, None, :] != A
    per_item_weight_matrix = np.abs(M - A)
    d_matrix = 1 - np.diag(np.ones((n)))
    # [i1,i2,j] for i1 <> i2 is the weight i2 gets on item j (high if opinion is SP). 0 on the diagonal i1=i2.
    per_item_weight_tensor = per_item_weight_matrix[None, :, :] + np.zeros(n)[:, None, None]
    per_item_weight_tensor = np.multiply(per_item_weight_tensor, d_matrix[:, :, None])

    # [i1,i2,j] is the `weighted similarity' of i1 and i2 on item j (0 if disagree, high if agree if i2 is SP)
    per_item_w_dist_tensor = np.multiply(per_item_diff_tensor, per_item_weight_tensor)

    # [i1,i2] is the total weight of i2 over all items
    sum_weights_ii = np.nansum(per_item_weight_tensor, 2)

    # [i1,i2] is the total weight of i2 on items where disagrees with i1
    sum_w_dist_ii = np.nansum(per_item_w_dist_tensor, 2)

    pairwise_proxy_scores_ii = np.divide(sum_w_dist_ii, sum_weights_ii)
    proxy_scores_ii = np.nanmean(pairwise_proxy_scores_ii, 1)

    # [i,j] is the total weight of anyone but i on item j
    sum_weights = np.nansum(per_item_weight_tensor, 1)

    # [i,j] is the total weight of all others that disagree with i on item j  (same for all workers with same answer)
    sum_w_dist = np.nansum(per_item_w_dist_tensor, 1)
    #

    # [i,j] is the average disagreement with i on item j (in [0,1])
    per_item_proxy_scores = np.divide(sum_w_dist, sum_weights)
    per_item_proxy_scores = np.multiply(per_item_proxy_scores, nan_map)

    # [i] is the average disagreement with i (in [0,1]) over all items
    proxy_scores = np.nanmean(per_item_proxy_scores, 1)
    # iteratively compute weights
    if SP_variant == "per_item":
        pass
    elif SP_variant == "pairwise":
        proxy_scores = proxy_scores_ii
        iter_weights = np.ones(n)
        max_iterations = iterations
        for iter in range(max_iterations):
            old_weights = iter_weights
            proxy_scores = compute_weighted_average_pairwise(pairwise_proxy_scores_ii, iter_weights)
            iter_weights = max(proxy_scores) - proxy_scores
            if old_weights is not None and np.linalg.norm(iter_weights - old_weights) < Constants.convergence_limit:
                break
    elif SP_variant == "use_kappa_for_proxy":
        worker_mean_answer = np.nanmean(A, 1)
        exp_disagreement_matrix = 2 * np.outer(worker_mean_answer, 1 - worker_mean_answer)
        exp_disagreement_vector = np.nanmean(exp_disagreement_matrix, 1)
        # kappa_matrix = np.divide(pairwise_proxy_scores_ii - exp_disagreement_matrix, 1 - exp_disagreement_matrix)
        proxy_scores = (1 + (proxy_scores_ii - exp_disagreement_vector)) / 2

    #     exp_disagreement_tensor = exp_disagreement_matrix[:, :, None] + np.zeros(m)[None, None, :]
    #     per_item_diff_tensor = np.divide(per_item_diff_tensor - exp_disagreement_tensor, 1 - exp_disagreement_tensor)

    estimated_fault = estimate_fault_from_proxy(proxy_scores, PTD_variant, mu_hat, Lambda, UR, possible_answers)
    weights = compute_weight_from_fault(estimated_fault, mode, positive, normalize, weight_transform, possible_answers)
    if SP_aggregate:
        # compare weighted support of the answer to its meta score (assume normalize weights):
        answers = np.zeros(m)
        for item_j in range(m):
            support_for_positive_answer = np.nansum(np.multiply(weights, A[:, item_j]))
            meta_support_for_positive_answer = np.nanmean(M[:, item_j])
            support_for_negative_answer = np.nansum(np.multiply(weights, 1 - A[:, item_j]))
            if support_for_positive_answer / (
                    support_for_positive_answer + support_for_negative_answer) > meta_support_for_positive_answer:
                answers[item_j] = 1
    else:
        # use same worker weights to all items, ignore meta scores:
        answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)

    return answers
