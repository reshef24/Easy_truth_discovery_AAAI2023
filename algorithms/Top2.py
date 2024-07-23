import networkx as nx
import numpy as np
import copy

import pandas as pd
from scipy.stats import binom, norm
from functools import partial

from utils.helpers import x_choose_2, compile_param_list, get_pairwise_distances_ranking, get_pairwise_distances
from utils.aggregations import aggregate
from typing import Dict

"""
Based on the pseudo-code in: Kawase et. al, "Graph Mining Meets Crowdsourcing: Extracting
 Experts for Answer Aggregation"

Link: https://www.ijcai.org/proceedings/2019/0177.pdf

INPUT: Any distance-based data
PARAMS:
      - mode: use top2 or experts answer aggregation
"""

# see IPTD() for explanation of arguments
def Top2(data:pd.DataFrame, 
           gt, 
           mode, 
           pairwise_distance_function,
           params, 
           variant,           # "experts" or "top2"
           voting_rule):
    n, m = data.shape
    if mode == "rankings":
        possible_answers = 2
        dist_matrix = get_pairwise_distances_ranking(data, gt, pairwise_distance_function)
    else:
        possible_answers = set(np.unique(data.values))
        dist_matrix = get_pairwise_distances(data, pairwise_distance_function)

    if mode == "categorical":
        p = ((1 - dist_matrix).sum() - n) / (2 * x_choose_2(n))  # dist matrix is already normalized in m
        similarity_matrix = Modes.Gamma.get(mode, calc_normal_distance_matrix)(dist_matrix * m, p, m)


    else:
        # workaround for general distance measures
        scaled_dist_matrix = dist_matrix / np.max(dist_matrix)
        similarity_matrix = 1 - scaled_dist_matrix

    experts = peeling(similarity_matrix, variant)
    weights = np.zeros(n)
    weights[experts] = 1 / len(experts)
    answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)
    return answers


def peeling(similarity_matrix, mode):
    G = nx.from_numpy_array(similarity_matrix)
    S = []
    nodes = []
    while G.nodes:
        S.append(copy.deepcopy(G))
        weighted_degree = dict(G.degree(weight='weight'))
        min_index = min(weighted_degree, key=weighted_degree.get)
        nodes.append(min_index)
        G.remove_node(min_index)

    if mode == Modes.Top2:
        return nodes[-3:]

    elif mode == Modes.Experts:
        max_k = float("-inf")
        experts = S[0]
        for s in S:
            weighted_degree = dict(s.degree(weight='weight'))
            k = min(weighted_degree.values())
            if k > max_k:
                experts = s.nodes
                max_k = k
    return list(experts)


def calc_binomial_distance_matrix(dist_matrix: np.array, p: float, m: int):
    # The cdf of # disagreements x = probability of # agreements is at least m - x
    vectorized_binom = np.vectorize(partial(binom.cdf, n=m, p=p))
    binomial_distance_matrix = - np.log(vectorized_binom(dist_matrix))
    return binomial_distance_matrix


def calc_normal_distance_matrix(dist_matrix: np.array, p: float, m: int):
    mu = p

    # Removing diagonal for sigma_square computation
    n = dist_matrix.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s1: object
    s0, s1 = dist_matrix.strides
    dist_matrix_no_diag = strided(dist_matrix.ravel()[1:], shape=(n - 1, n), strides=(s0 + s1, s1)).reshape(n, -1)
    sigma_square = dist_matrix_no_diag.std()

    # The cdf of #distance = probability of #similarity at most distance
    vectorized_binom = np.vectorize(partial(norm.cdf, scale=sigma_square, loc=mu))
    normal_distance_matrix = - np.log(vectorized_binom(dist_matrix))
    return normal_distance_matrix


class Modes:
    Experts = "experts"
    Top2 = "top2"
    Gamma = {"categorical": calc_binomial_distance_matrix,
             "continuous": calc_normal_distance_matrix}
