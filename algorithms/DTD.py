import numpy as np
import pandas as pd
from scipy.spatial import distance

from configurations.constants import Constants
from utils.aggregations import aggregate
from utils.helpers import bound, ranking_to_pairwise_q, compute_weight_from_fault
from utils.voting_rules import voting_rules_dict, run_vr
from utils.wquantile import wmedian

"""
IDTD = Iterative Distance-from-answer Truth Discovery. Iteratively computes the aggregated answer and re-estimates 
fault based on distance from these answers.
This is a "Folk algorithm" based on ideas occurring in multiple papers.

We provide separate implementations for the common data types: for continous with mean/median aggregation; for rankings; and for all others.
INPUT: Any distance-based data.  
"""
# Not mentioned in the paper.

def IDTD_mean(data: pd.DataFrame,
              iterations,   # number of iterations
              mode,     # "continuous"
              positive=False,
              normalize=True,
              weight_transform="log"):
    n = data.shape[0]
    weights = np.ones(n)
    c_lb = Constants.fault_min
    c_ub = Constants.fault_max
    A = data.values

    for iter in range(iterations):
        w_mean = np.average(A, axis=0, weights=weights)
        D = A - w_mean
        estimated_fault = np.mean(D ** 2, axis=1)
        estimated_fault = bound(c_lb, c_ub, estimated_fault)
        weights = compute_weight_from_fault(estimated_fault, mode, positive, normalize, weight_transform)

    answers = np.average(data, axis=0, weights=weights, returned=False)

    return answers


def DTD_median(data: pd.DataFrame):
    df_median = data.median(axis=0)
    k = data.shape[1]
    D = data - df_median
    distance_from_median = np.mean(D ** 2, axis=1)
    c_lb = Constants.fault_min
    c_ub = Constants.fault_max
    estimated_fault = bound(c_lb, c_ub, distance_from_median)
    weights = 1 / estimated_fault
    answers = np.zeros(k)
    for j in range(k):
        answers[j] = wmedian(data[j], weights)
    return answers


  # see IPTD() for explanation of arguments
def IDTD_ranking(data: pd.DataFrame,
                 gt_ranking,   # as in IPTD, the ground truth ranking is required only to translate the ranking data
                 iterations,
                 positive,
                 normalize,
                 weight_transform,
                 voting_rule, 
                 params):
    weights = None
    n = data.shape[0]
    estimated_fault = np.ones(n)
    VR_function = voting_rules_dict[voting_rule]

    for iter in range(iterations):
        last_weights = weights
        estimated_outcome = run_vr(data, gt_ranking, VR_function, weights, **params)
        estimated_outcome_pairwise = ranking_to_pairwise_q([estimated_outcome], gt_ranking)
        for i in range(n):
            estimated_fault[i] = distance.hamming(data.iloc[i][:], estimated_outcome_pairwise, w=None)
        estimated_fault = bound(Constants.fault_min_cat, Constants.fault_max_cat, estimated_fault)
        weights = compute_weight_from_fault(estimated_fault, "rankings", positive, normalize, weight_transform,
                                            possible_answers=2)
        if np.array_equal(weights, last_weights):
            break
    answers = run_vr(data, gt_ranking, VR_function, weights, **params)

    # return the actual number of iterations under 'alpha'
    return answers

 # see IPTD() for explanation of arguments
def IDTD(data: pd.DataFrame,
         gt,
         iterations,
         mode,
         voting_rule,
         dist_function,
         params,
         positive=True,
         normalize=True,
         weight_transform="simple",
         ):
    possible_answers = set(np.unique(data.values)) if mode != "rankings" else 2
    n = data.shape[0]
    weights = np.ones(n)
    for iter in range(iterations):
        old_weights = weights
        answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)
        distances = np.zeros(n)
        for i in range(n):
            if mode == "etchacell":
                distances[i] = dist_function(data.iloc[i][0], answers)
            else:
                distances[i] = dist_function(data.iloc[i][:], answers)

        fault = bound(Constants.fault_min_cat, Constants.fault_max_cat, distances)
        weights = compute_weight_from_fault(fault,
                                            mode,
                                            positive,
                                            normalize,
                                            weight_transform,
                                            possible_answers)
        if old_weights is not None and np.linalg.norm(weights - old_weights) < Constants.convergence_limit:
            break
    answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)

    return answers
