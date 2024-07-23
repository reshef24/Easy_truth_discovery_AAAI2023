import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.spatial import distance

from utils.aggregations import plurality
from utils.helpers import bound

"""
Eigen Vector-based truth discovery. Following the [Parisi et al. PNAS'14] paper.
We implement the SML algorithm for *balanced* data (with a single parameter per worker)
synthetic data must be balanced!

There are various ways to compute the main diagonal. The one from the paper is 'lsq' with 2 standard deviations.
INPUT: Binary data only.
PARAMS: 
        - diagonal: how to estimate the diagonal of the covariance matrix.
        - stds: how many standard deviations to use for the diagonal estimation.
        - positive: whether to clip the weights to be positive or not.
        - flip: whether to flip the leading eigenvector to be positive or not.
"""


def EVTD(data: pd.DataFrame, gt = None, positive=False, diagonal='lsq', stds=2, flip=True):
    df_nadler = data.replace(0, -1)  # Change zeros to -1
    n, m = data.shape
    possible_answers = len(np.unique(data.values)) # fixed it from being set to being number of unique values

    q_ii_hat = np.zeros(n)
    Q = np.cov(df_nadler)  # Covariance matrix
    df_Q = pd.DataFrame(Q)
    vQ = build_vQ(df_nadler, df_Q)
    Q_valid_indices = np.absolute(Q) >= stds * vQ ** 0.5
    for i in range(n):
        Q_valid_indices.loc[i, i] = 0
    num_of_valid = (int)(np.sum(np.sum(Q_valid_indices)) / 2)  # only upper triangle matters

    if diagonal == "lsq":
        if num_of_valid >= n:  # otherwise this is under-determined
            C = np.zeros(num_of_valid)
            A = np.zeros(shape=(num_of_valid, n))
            current_row = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if Q_valid_indices[i][j]:
                        if Q[i][j] != 0:
                            C[current_row] = np.log(np.absolute(Q[i][j]))
                            A[current_row][i] = 1
                            A[current_row][j] = 1
                        current_row += 1

            t_hat = lstsq(A, C, rcond=None)[0]
            q_ii_hat = np.exp(2 * t_hat)

    if diagonal == "average" or diagonal == "average_then_sqr":
        ## use the average of all off-diagonal values of this row
        for i in range(n):
            val = 0
            count = 0
            for j in range(n):
                if j != i and Q_valid_indices[i][j]:
                    val = val + Q[i][j]
                    count = count + 1
            if count > 0:
                q_ii_hat[i] = val / count
        if diagonal == "average_then_sqr":
            ## Then divide by mu and raise to power 2
            mu_hat = np.average(q_ii_hat)
            if mu_hat > 0:
                q_ii_hat = np.square(q_ii_hat) / mu_hat

    if diagonal == "sqr_collect":
        ## estimate diagonal from all triplets by collecting
        for i in range(n):
            nominator = 0
            denominator = 0
            for j in range(n):
                if j == i or not Q_valid_indices[i][j]:
                    continue
                for k in range(n):
                    if k == j or k == i or not Q_valid_indices[k][j] or not Q_valid_indices[k][i]:
                        continue
                    nominator += Q[i, j] * Q[i, k]
                    denominator += Q[j, k]
            if denominator != 0:
                q_ii_hat[i] = nominator / denominator

    if diagonal == "sqr_then_average":
        ## estimate diagonal from all triplets by averaging
        for i in range(n):
            quads = []
            for j in range(n):
                if j == i or not Q_valid_indices[i][j]:
                    continue
                for k in range(n):
                    if k == j or k == i or not Q_valid_indices[k][j] or not Q_valid_indices[k][i]:
                        continue
                    if Q[j][k] != 0:
                        quads.append(Q[i, j] * Q[i, k] / Q[j, k])
            if len(quads) > 0:
                q_ii_hat[i] = np.average(quads)

    if diagonal == "oracle":
        ## use the true fault levels (squared)
        for i in range(n):
            true_competence_i = 1 - distance.hamming(data.iloc[i][:], gt, w=None)
            q_i = true_competence_i * 2 - 1
            q_ii_hat[i] = np.square(q_i)

    np.fill_diagonal(Q, bound(0, 1, q_ii_hat))

    if diagonal == "SVD":
        ## keep the empirical q_ii
        Q = df_nadler.dot(df_nadler.T) / m

    w, v = np.linalg.eigh(Q)
    max_ind = np.argmax(w)
    leading_vec = v[:, max_ind]
    # leading_vec = power_method_square_matrix(Q,accuracy=n*0.00000001)
    if flip:
        if np.sum(leading_vec) < 0:
            leading_vec = -leading_vec
    eigenvalue = w[max_ind]
    # np.linalg.norm(np.matmul(Q,leading_vec))

    weights = leading_vec.reshape(1, -1)
    if positive:
        weights = np.maximum(0, weights)
    answers = plurality(df_nadler.T, possible_answers, weights, signed=True)
    answers = (answers + 1) / 2

    ## this is just for the output:
    normalized_leading_vec = leading_vec * np.sqrt(eigenvalue)
    estimated_fault = (1 - normalized_leading_vec) / 2

    return answers

def build_vQ(df_nadler, Q):
    gamma = np.array(df_nadler.mean(axis=1)).reshape(1, -1)
    m = df_nadler.shape[1]
    vQ = np.matmul((1 - gamma ** 2).T, (1 - gamma ** 2)) / (m - 1) + \
         (Q / m) * (4 * np.matmul(gamma.T, gamma) - Q * (m - 2) / (m - 1))
    return vQ


def calc_real_Q(df: pd.DataFrame, gt, dist_func):
    pi = []
    for row in df.iterrows():
        dist = dist_func(pd.Series(row)[1], pd.Series(gt.T))
        pi.append(dist)
    pi = np.array(pi).reshape(-1, 1)
    v = 2 * pi - 1
    v = v / np.sqrt(sum(v ** 2))

    Q = np.matmul(v, v.T)

    return Q
