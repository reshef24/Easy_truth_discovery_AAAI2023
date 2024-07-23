"""
INPUT: 
    - data:
    - gt: ground truth
PARAMS:
    - mode: categorical, rankings, continuous
    - distance_function: distance function to use
    - params: parameters for the distance function
    - positive: whether to use positive fault or not
    - normalize: whether to normalize the fault or not
    - weight_transform: which weight transformation to use
    - voting_rule: which voting rule to use
"""
import numpy as np
import pandas as pd

from configurations.constants import Constants
from utils.aggregations import aggregate
from utils.helpers import compute_weight_from_fault, bound_single

from measures.vector_distance_functions import hamming_binary


def Oracle(data: pd.DataFrame,
           gt,
           mode,
           distance_function,
           params,
           positive=True,
           normalize=True,
           weight_transform="simple",
           voting_rule = None):
    n = data.shape[0]
    true_fault = np.zeros(n)
    for i in range(n):
        data_i = data.iloc[i][:]
        true_fault[i] = compute_fault_supervised(mode, gt, data_i, distance_function)

    result_dict = af_external_fault(data, gt, positive, normalize, weight_transform, mode,
                                    true_fault, params, voting_rule)
    return result_dict

# weighted plurality based on fault, where fault is given externally and perturbed by noise


def af_external_fault(data,
                      gt,
                      positive,
                      normalize,
                      weight_transform,
                      mode,
                      external_fault,
                      params,
                      voting_rule):
    possible_answers = set(np.unique(data.values))
    weights = compute_weight_from_fault(external_fault, mode, positive, normalize, weight_transform, possible_answers=2)
    answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)
    return answers


def compute_fault_supervised(mode, gt, data_i, distance_function):
    if mode == "rankings":
        # Data is converted to binary
        distance_function = hamming_binary

    fault = distance_function(data_i, gt)
    if mode in ["categorical", "rankings"]:
        fault = bound_single(Constants.fault_min_cat, Constants.fault_max_cat, fault)
    elif mode == "continuous":
        fault = bound_single(Constants.fault_min, Constants.fault_max, fault)
    return fault
