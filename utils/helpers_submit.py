from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd
from scipy import special

from configurations.constants import Constants


def get_param_range(min, max, count):
    if count <= 2:
        return range(min, max + 1)


def bound(lower, upper, p):
    """ bound p from above and below, p=0.5 will produce weight=0"""
    for i in range(len(p)):
        p[i] = bound_single(lower, upper, p[i])
    return p


def bound_single(lower, upper, p):
    """ bound p from above and below, p=0.5 will produce weight=0"""
    if p < lower:
        v = lower
    elif p > upper:
        v = upper
    else:
        v = p
    return v


# select a random index according to weights. Normalize to positive weights only.
def w_random(n, weights=None):
    if weights is None:
        weights = np.ones(n) / n
    else:
        pos_w = np.maximum(0, weights)
        weights = pos_w / sum(pos_w)
    return np.random.choice(range(n), p=weights)


def ranking_to_pairwise_q(ranking_data, gt) -> (list, list):
    k_ranking = len(ranking_data[0])
    k = int(special.binom(k_ranking, 2))
    ans = np.full((len(ranking_data), k), fill_value=-1, dtype=None, order='C')
    gt = list(map(int, gt))
    for idx, row in enumerate(ranking_data):
        j_inx = 0
        for i in range(k_ranking):
            for j in range(i + 1, k_ranking):
                if row.index(gt[i]) < row.index(gt[j]):
                    ans[idx, j_inx] = 1
                else:
                    ans[idx, j_inx] = 0
                j_inx = j_inx + 1
    return ans


def get_all_ranking_as_pairwise(ranking_gt):
    all_ranking = permutations(ranking_gt)
    all_ranking = [list(map(int, x)) for x in list(all_ranking)]
    all_ranking_as_pairwise = ranking_to_pairwise_q(all_ranking, ranking_gt)
    return all_ranking, all_ranking_as_pairwise


def x_choose_2(x, y=2):
    return factorial(x) / (factorial(x - y) * factorial(y))


def compute_option_number_from_sample_data(s_data: pd.DataFrame) -> int:
    arr = np.array(s_data).reshape(1, -1)
    return len(np.unique(arr[~pd.isna(arr)]))


# runs until reaching iteration limit OR accuracy limit
def power_method_square_matrix(A, accuracy=0.0000001, max_iterations=100):
    z = np.random.rand(A.shape[0])
    for i in range(max_iterations):
        next_z = np.matmul(A, z)
        next_z = next_z / np.linalg.norm(next_z)
        if np.linalg.norm(next_z - z) < accuracy:
            break
        z = next_z
    return next_z


def compile_param_list(params):
    str = ""
    for param in params:
        str += "{}:{}; ".format(param, params[param])
    return str


def compute_excluded_distances_from_mean(df, weights=None):
    n = df.shape[0]
    distances = np.zeros(n)
    A = df.values
    R = range(n)
    for i in range(n):
        mask = list(R[:i]) + list(R[i + 1:])
        excluded_A = A[mask, :]
        excluded_answers = np.average(excluded_A, axis=0, weights=weights)
        distances[i] = np.mean((A[i, :] - excluded_answers) ** 2)
    return distances


def entropy(x):
    return max(np.absolute(x)) / sum(np.maximum(x, 0))


def compute_weight_from_fault(fault, mode, w_params, possible_answers=2):
    n = len(fault)
    positive = w_params.get("positive", False)
    normalize = w_params.get("normalize", True)

    if mode in ["categorical", "rankings"]:
        transform_method = w_params.get("weight_transform", "log")
        competence = 1 - fault
        if transform_method == "log":
            weights = np.log(competence * (possible_answers - 1) / (1 - competence))
        elif transform_method == "simple":
            weights = competence - 1 / possible_answers
        else:
            Exception("weight method not specified")
    elif mode == "continuous":
        weights = 1 / fault
    else:
        weights = 1 - fault
    if positive:
        weights = np.maximum(weights, 0)
    if np.isnan(sum(weights)):
        print("NaN here")
    if weights is None or np.sum(weights) < 0 or np.sum(np.sign(weights)) <= 0 or np.isnan(np.sum(weights)):
        return np.ones(n) / n
    else:
        if normalize:
            return weights / np.sum(np.maximum(0, weights))
        else:
            return weights

# estimate individual fault (or competence) from the proxy score according to the variant
def estimate_fault_from_proxy(proxy_score, variant, est_params):
    mu_hat = est_params.get('mu_hat', np.mean(proxy_score) / 2)
    n = len(proxy_score)
    if min(proxy_score) < 0:
        raise Exception('negative proxy score')


    if variant == "PP":
        estimated_competence = proxy_score
        return estimated_competence
    elif variant == "AWG":
        Lambda = est_params.get('Lambda', 0)
        UR = est_params.get('unit_regularizer', False)
        if UR:
            # penalty for fault far from 1
            estimated_fault = 2 * (n - 1) / (2 * n + Lambda - 4) * proxy_score - 2 * (4 * n * (n - 1) * mu_hat + 1) / (
                    (4 * n + Lambda - 4) * (2 * n + Lambda - 4))
        else:
            # penalty for fault far from 0
            estimated_fault = 2 * (n - 1) / (2 * n + Lambda - 4) * proxy_score - (
                    (8 * n * (n - 1)) / ((4 * n + Lambda - 4) * (2 * n + Lambda - 4))) * mu_hat
        estimated_fault = bound(Constants.fault_min, Constants.fault_max, estimated_fault)
    elif variant in ["IER","IER_BIN"]:
        possible_answers = est_params.get('possible_answers', 2)
        theta = 1 / (possible_answers - 1)
        estimated_fault = (proxy_score - mu_hat) / (1 - (1 + theta) * mu_hat)
        estimated_fault = bound(Constants.fault_min_cat, Constants.fault_max_cat, estimated_fault)
    else:  # variant PD
        estimated_fault = proxy_score
        estimated_fault = bound(Constants.fault_min, Constants.fault_max, estimated_fault)

    return estimated_fault

# return the weighted row average of the pairwise distance (or proximity) matrix, ignoring the main diagonal.
def compute_weighted_average_pairwise(pairwise_matrix, weights):
    n = pairwise_matrix.shape[0]
    proxy_score = np.ones(n)

    # override and use equal weights if weights are mostly negative
    if weights is None or np.sum(weights) < 0 or np.sum(np.sign(weights)) <= 0 or np.isnan(np.sum(weights)):
        proxy_score = pairwise_matrix.sum(axis=0) / (n - 1)
    else:
        sum_w = np.sum(weights)

        for i in range(n):
            sum_w_except_i = (sum_w - weights[i])
            if sum_w_except_i <= 0:
                w_norm = np.ones(n) / (n - 1)
            else:
                w_norm = weights / sum_w_except_i
            w_norm[i] = 0
            proxy_score[i] = np.dot(pairwise_matrix[i, :], w_norm)

    proxy_score = np.squeeze(np.asarray(proxy_score))
    return proxy_score