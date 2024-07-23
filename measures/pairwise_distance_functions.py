import numpy as np
import pandas as pd
import measures.vector_distance_functions
import measures.gleu_distance as gd
from itertools import combinations_with_replacement

from sklearn import metrics
from scipy import stats


def hamming_binary_pairwise(df):
    """ computes the un-normalized Hamming distance between each couple of workers """
    """ not for partial data """
    if  isinstance(df, pd.DataFrame):
        A = df.values
    else:
        A = df
    return 2 * np.inner(A - 0.5, 0.5 - A) + A.shape[1] / 2


def gleu_distance_pairwise(df):
    return gd.gleu_all_pairs(df)


def jaccard_distance_pairwise(df):
    """
    Every row is a single reported answer. Every column is an element. The order of columns of each row is arbitrary.
    """
    answers_num = len(df)
    answer_list = [df.iloc[i][0] for i in range(answers_num)]

    indices = list(combinations_with_replacement(range(answers_num), 2))

    result = np.zeros(shape=(answers_num, answers_num))
    for i, j in indices:
        value1 = answer_list[i]
        value2 = answer_list[j]
        distance = measures.vector_distance_functions.jaccard_distance(value1, value2)
        result[i, j] = result[j, i] = distance
    return result


def hamming_general_pairwise(df):
    A = df.values
    return (A[:, None, :] != A).sum(2)


def square_euclidean(df, y=None):
    return metrics.pairwise.euclidean_distances(df, Y=y, squared=True)


def square_euclidean_no_zero(df):
    df_frame = df
    answers = df_frame.shape[0]
    indices = list(combinations_with_replacement(range(answers), 2))

    result = np.zeros(shape=(answers, answers))
    distance = None
    for i, j in indices:
        value1 = df_frame.loc[i]
        value2 = df_frame.loc[j]
        distance = measures.vector_distance_functions.square_euclidean_no_zero(value1, value2, pairwise=True)
        result[i, j] = result[j, i] = distance

    return result


def square_probability(df, y=None):
    dft = np.log(np.divide(np.maximum(0.001, df), np.maximum(0.001, 1 - df)))
    if y is not None:
        yt = np.log(np.divide(np.maximum(0.001, y), np.maximum(0.001, 1 - y)))
    else:
        yt = None
    return metrics.pairwise.euclidean_distances(dft, Y=yt, squared=True)


def l1(df, y=None):
    """Normalizing in the sum"""
    result = metrics.pairwise.manhattan_distances(df, Y=y)
    return result


def spearman_rank_correlation(df: pd.DataFrame):
    return 1 - ((stats.spearmanr(a=df, axis=1).correlation + 1) / 2)


def footrule(df):
    return metrics.pairwise_distances(X=df.to_numpy(),
                                      metric=measures.vector_distance_functions.footrule_distance)


def compute_pairwise_distances_discrete_per_question(df):
    """ computes the un-normalized Hamming/Euclidean distance between each couple of workers """
    """ the result is a 3D array, where in each layer a single question is omitted """
    """ not for partial data """
    A = df.values
    k = A.shape[1]
    dist_per_question = (A[:, None, :] != A)
    dist_all = dist_per_question.sum(2)
    dist_repeated = np.repeat(dist_all[:, :, np.newaxis], k, axis=2)
    return dist_repeated - dist_per_question


def compute_pairwise_distances_discrete(df, binary):
    """ computes the un-normalized Hamming/Euclidean distance between each couple of workers """
    """ not for partial data """
    A = df.values
    if binary:  # slightly faster
        return (2 * np.inner(A - 0.5, 0.5 - A) + A.shape[1] / 2)
    else:
        return (A[:, None, :] != A).sum(2)
