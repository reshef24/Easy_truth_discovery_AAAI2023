import ast
import pandas as pd
import numpy as np
import measures.gleu_distance as gd

from scipy.stats import spearmanr
from utils.helpers import x_choose_2, str_to_coord_list

MIN_ZERO_DIST = 0.25


def footrule_normalizer(x):
    """https://mikespivey.wordpress.com/2014/01/20/the-maximum-value-of-spearmans-footrule-distance/"""
    m = len(x)
    return (2 * m ** 2) + 2 * m * (m % 2)


def square_euclidean(x: np.ndarray, y: np.ndarray):
    distance = (np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)) / len(x)
    return distance


# assumes answers are in [0,100].
# an answer of `0' can be treated as an uncertain answer, especially if there are many zeros.
def square_euclidean_no_zero(x: np.ndarray, y: np.ndarray, pairwise=False):
    distance = 0
    m = len(x)
    x_zeros = np.sum(x == 0)
    y_zeros = np.sum(y == 0)
    x_zero_dist = 50 * x_zeros / m
    y_zero_dist = 50 * y_zeros / m
    if not pairwise:
        y_zero_dist = 0
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0:
            distance = distance + (x_zero_dist + y_zero_dist) ** 2
        elif x[i] == 0:
            distance = distance + (x_zero_dist + y[i]) ** 2
        elif y[i] == 0:
            distance = distance + (x[i] + y_zero_dist) ** 2
        else:
            distance = distance + (x[i] - y[i]) ** 2
    distance = np.sqrt(distance / m)
    return distance


def square_probability(x: np.ndarray, y: np.ndarray):
    tx = np.log(np.divide(np.maximum(0.001, x), np.maximum(0.001, 1 - x)))
    ty = np.log(np.divide(np.maximum(0.001, y), np.maximum(0.001, 1 - y)))
    return square_euclidean(tx, ty)


def hamming_binary(x: np.ndarray, y: np.ndarray):
    x_ne_y = x != y
    return np.average(x_ne_y)


"""Using only the name to indicate nicely for the pairwise distance"""


def hamming_general(x: np.ndarray, y: np.ndarray):
    x_ne_y = x != y
    return np.average(x_ne_y)


def l1(x: np.ndarray, y: np.ndarray):
    result = np.sum(np.abs(np.array(x) - y))
    return result / len(x)


def kendall_tau_distance(order_a, order_b):
    order_a = [int(i) for i in list(order_a)]  # list of str -> list of int
    order_b = [int(i) for i in list(order_b)]  # list of str -> list of int

    distance = 0
    # for x, y in pairs:
    for x in order_a:
        for y in order_a:
            a = order_a.index(x) - order_a.index(y)
            b = order_b.index(x) - order_b.index(y)
            if a * b < 0:
                distance += 1
    return distance / 2 / x_choose_2(len(order_a))


"""
# each is a list of strings. If input is a string it is converted to list
"""


def jaccard_distance(vector_a, vector_b):
    if len(vector_a) == 1 and len(vector_b) == 1:
        # FIXME: ugly hack to handle single-column datasets
        return jaccard_distance(vector_a[0], vector_b[0])
    if isinstance(vector_a, str):
        vector_a = str_to_coord_list(vector_a, items_as_str=True)
    if isinstance(vector_b, str):
        vector_b = str_to_coord_list(vector_b, items_as_str=True)
    set_a = set(vector_a)
    set_b = set(vector_b)
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return 1 - len(intersection) / len(union)


def spearman_rank_corr(x, y):
    """Correlation is from -1 to 1"""
    return 1 - ((spearmanr(x, y).correlation + 1) / 2)


def footrule_distance(x: np.array, y: np.array):
    """
    finds the indices of x values in y
    calculate the sum of abs difference
    :param x:
    :param y:
    :return:
    """
    y_location = np.where(y.reshape(y.size, 1) == x)[1]
    abs_diff = np.abs(y_location - range(len(x)))
    return np.sum(abs_diff) / footrule_normalizer(x)


def gleu_distance(x: np.array, y: np.array):
    return gd.gleu_distance(x, y)
