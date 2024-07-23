import numpy as np
import pandas as pd

from configurations.constants import Constants
from utils.aggregations import aggregate
from utils.helpers import compute_weight_from_fault, \
    estimate_fault_from_proxy, compute_weighted_average_pairwise, get_pairwise_distances, get_pairwise_distances_ranking

"""
[Iterative] Proximity-based Truth Discovery. 
variants:
* pairwise_proximity (PP): competence_j = avg(p(i,j)); weight_j = competence_j
* pairwise_distance (PD):  fault_j = avg(d(i,j));  weight_j = max(fault) - fault_j 
* AWG     (only in continuous mode)
* IER     (only in categorical mode)
* IER_BIN  (only in binary/ranking mode)
* IER_SP  (only in binary mode)

INPUT: Any distance-based data 
"""


def IPTD(data: pd.DataFrame,
         voting_rule,  # [str] specifies which voting rule is used for aggregation. Allowed rules depend on mode.
                       # any mode: "random_dictator" or "best_dictator"
                       # continuous mode: "mean" or "median"
                       # categorical mode: ignored (always use plurality)
                       # ranking mode: any of the functions starting with vr_  on file  utils/voting_rules.py
         iterations,  # number of iterations for the iterative version
         iter_weight_method,  # determines how weights are updated in each iteration:  
                              # "max":  weight is max(proxy score) - proxy score 
                              # "1":   weight is 1 - proxy score 
                              # "from_estimate": apply domain-specific fault estimate and and fault-to-weight functions
         variant,           # [str] variants: 
                            # * pairwise_proximity (PP): competence_j = avg(p(i,j)); weight_j = competence_j
                            # * pairwise_distance (PD):  fault_j = avg(d(i,j));  weight_j = max(fault) - fault_j 
                            # * AWG     (only in continuous mode)
                            # * IER     (only in categorical mode)
                            # * IER_BIN  (only in binary/ranking mode)
                            # * IER_SP  (only in binary mode)
         mode,            # the data type: "rankings" / "categorical" / "continuous" etc     
         pairwise_distance_function,   # [function] one of the distance functions listed in algorithms/__init__.py under pair_distance_functions
         params = None,          # [dictionary] any additional parameters that we want to pass to the voting_rule (used in rankings mode)
         mu_hat = None,          # an external estimate of the average proxy score if not given: np.mean(proxy_score) / 2
         Lambda = None,          # regularization factor for the AWG variant:  
                                    # 0 is the MLE  (no regularization)
                                    # 4 is equivalent to the PD variant
         weight_transform="simple",  # [str] only relevant for categorical mode: 
                                     # "log": use Grofman weights
                                     # "simple" use linear approximation of Grofman weights
         normalize=True,    # normalize weights so that all positive weights sum to 1
         gt = None,    # a vector of the ground truth answers
         positive=True):    # set negative weights to 0
    
    PTD_variant = variant
    max_iterations = iterations

    if mode == "rankings":
        possible_answers = 2  # since rankings are translated to binary vectors
        pairwise_matrix = get_pairwise_distances_ranking(data, gt, pairwise_distance_function)
    else:
        possible_answers = len(set(np.unique(data.values)))
        pairwise_matrix = get_pairwise_distances(data, pairwise_distance_function)

    # iteratively compute weights
    iter_weights = None
    for iter in range(max_iterations):
        old_weights = iter_weights
        proxy_scores = compute_weighted_average_pairwise(pairwise_matrix, iter_weights)
        if PTD_variant == "PP":
            iter_weights = proxy_scores
        elif iter_weight_method == "max":
            iter_weights = max(proxy_scores) - proxy_scores
        elif iter_weight_method == "1":
            iter_weights = 1 - proxy_scores
        elif iter_weight_method == "from_estimate":
            iter_estimated_fault = estimate_fault_from_proxy(proxy_scores, variant, mu_hat, Lambda,
                                                             possible_answers)
            iter_weights = compute_weight_from_fault(iter_estimated_fault, mode, positive, normalize, weight_transform,
                                                     possible_answers)

        if old_weights is not None and np.linalg.norm(iter_weights - old_weights) < Constants.convergence_limit:
            break
    
    #### TODO
    ####
    estimated_fault = estimate_fault_from_proxy(proxy_scores, PTD_variant, possible_answers, mu_hat, Lambda)
    weights = compute_weight_from_fault(estimated_fault, mode, positive, normalize, weight_transform, possible_answers)
    answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)
    return answers


