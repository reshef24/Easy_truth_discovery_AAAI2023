import random
from pathlib import Path
import pandas as pd
import numpy as np
import csv
from scipy.stats import multinomial,truncnorm
import math
from scipy.spatial import distance


def _icc_exponent(theta, a, b):
    return np.exp(-a * (theta[:, None] - b[None, :]))


def icc(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
    """Item Response Theory four-parameter logistic function [Ayala2009]_, [Magis13]_:
    .. math:: P(X_i = 1| \\theta) = c_i + \\frac{d_i-c_i}{1+ e^{a_i(\\theta-b_i)}}
    :param theta: the individual's proficiency value. This parameter value has
                  no boundary, but if a distribution of the form :math:`N(0, 1)` was
                  used to estimate the parameters, then :math:`-4 \\leq \\theta \\leq
                  4`.
    :param a: the discrimination parameter of the item, usually a positive
              value in which :math:`0.8 \\leq a \\leq 2.5`.
    :param b: the item difficulty parameter. This parameter value has no
              boundaries, but it is necessary that it be in the same value space
              as `theta` (usually :math:`-4 \\leq b \\leq 4`).
    :param c: the item pseudo-guessing parameter. Being a probability,
              :math:`0\\leq c \\leq 1`, but items considered good usually have
              :math:`c \\leq 0.2`.
    :param d: the item upper asymptote. Being a probability,
              :math:`0\\leq d \\leq 1`, but items considered good usually have
              :math:`d \\approx 1`.
    """
    return c + ((d - c) / (1 + _icc_exponent(theta, a, b)))


def write_data_to_file(n, s, gt, base_name, write_to_dir, levels=None):
    file_name = base_name + '.csv'
    gt_name = base_name + '_GT' + '.csv'
    p_name = base_name + '_P' + '.csv'
    write_to = (write_to_dir / file_name)  # .resolve()
    write_to_GT = (write_to_dir / gt_name)  # .resolve()
    write_to_P = (write_to_dir / p_name)  # .resolve()
    # # ---------------------------------------------------------- #
    # write data:
    with open(str(write_to), mode='w', newline='') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(n):
            employee_writer.writerow(s.loc[i, :])

    # write ground truth:
    with open(str(write_to_GT), mode='w', newline='') as GT_file:
        employee_writer = csv.writer(GT_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(gt)

    # write true competence:
    if levels is not None:
        with open(str(write_to_P), mode='w', newline='') as P_file:
            employee_writer = csv.writer(P_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow(levels)


def write_ranking_data_to_file(n, s, gt, base_name, write_to_dir, levels=None):
    file_name = base_name + '.csv'
    gt_name = base_name + '_GT' + '.csv'
    write_to = (write_to_dir / file_name)
    write_to_GT = (write_to_dir / gt_name)
    with open(write_to, 'w', newline='') as out:
        csv_out = csv.writer(out)
        for row in s:
            csv_out.writerow(row)

    with open(write_to_GT, 'w', newline='') as out2:
        csv_out = csv.writer(out2)
        csv_out.writerow(gt)


def create_categorial_iid(n, m, possible_outcomes, c_levels, dist_string, difficulty=None, balance = False):
    print("creating " + dist_string)
    if balance:
        gt = pd.Series(list(range(possible_outcomes)) * (int)(m/possible_outcomes+1))  # truth
        gt = gt[0:m]
        random.shuffle(gt)
    else:
        gt = pd.Series([0] * m)
    ## uniform in range [lb,ub]
    base_name = 'CAT{}_m{}_n{}_{}'.format(possible_outcomes, m, n, dist_string)
    # ---------------------------------------------------------- #
    w = np.log(c_levels / (1 - c_levels))
    if difficulty is None:
        difficulty = np.zeros(m)
        P_matrix = icc(w, 1, difficulty, 0)
    else:
        P_matrix = icc(w, 1, difficulty, 1 / possible_outcomes)
    s = np.empty(shape=[n, m])
    for j in range(m):
        for i in range(n):
            pval = list([P_matrix[i, j]])
            q_i = ((1 - P_matrix[i, j]) / (possible_outcomes - 1))
            q_list = [q_i] * (possible_outcomes - 1)
            q_list.insert(gt[j],pval[0])
            rv = multinomial(1, q_list)
            s_ij = rv.rvs(size=1, random_state=None)[0]
            s[i, j] = np.where(s_ij == 1)[0]
    s = pd.DataFrame(s)
    return s, gt, base_name


def create_continuous_iid(n, m, f_levels, dist_string, gt =None, difficulty=None):
    base_name = 'CONT_m{}_n{}_{}'.format(m, n, dist_string)
    if gt is None:
       gt = np.zeros(m)
    if difficulty is None:
        difficulty = np.ones(m)
    P_matrix = f_levels[:, None] * difficulty[None, :]
    s = np.empty(shape=[n, m])
    for j in range(m):
        # mu_j = np.random.normal(j, 1)
        # gt[j] = mu_j
        for i in range(n):
            s[i, j] = np.random.normal(gt[j], f_levels[i])
    s = pd.DataFrame(s)
    return s, base_name


def main_cat_binary():
    # TODO: set parameters:
    possible_outcomes = 2
    k = 2000  # questions
    n = 2000  # workers
    lb, ub = 0.4, 0.7
    write_to_dir = Path.cwd()  / 'synthetic'
    #
    # # ---------------------------------------------------------- #
    # # ## uniform in range [lb,ub]
    # # p = np.random.uniform(lb, ub, n)  # competence
    # # dist_name = "U"
    # c_levels = np.random.uniform(lb, ub, n)  # competence
    # difficulty = None  # np.random.normal(0,2,k)
    # dist_string = "U({},{})".format(lb, ub)
    # s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    # write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
    #
    # # ---------------------------------------------------------- #
    # # triangular with mode = chance guess
    # center = 1 / possible_outcomes
    # c_levels = np.random.triangular(lb,center, ub, n)  # competence
    # difficulty = None  # np.random.normal(0,2,k)
    # dist_string = "T({},{},{})".format(lb,center, ub)
    # s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    # write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
    #
    # # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    #
    #     ## Normal distribution with STD (ub-lb)/2
    #     center = (ub+lb)/2
    #     std = np.round((ub-lb)/2,2)
    # lb = (0-center)/std
    # ub = (1-center)/std
    # c_levels = truncnorm.rvs(
    #     lb, ub, loc=center, scale=std, size=n)
    #     dist_string = "N({},{})".format(center, std)
    #     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    #     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

#     center = 0.65
#     difficulty = None  # np.random.normal(0,2,k)
#
#     std = 0.15
#     lb = (0-center)/std
#     ub = (1-center)/std
#     c_levels = truncnorm.rvs(
#         lb, ub, loc=center, scale=std, size=n)
#     dist_string = "N({},{})".format(center, std)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#     center = 0.75
#     difficulty = None  # np.random.normal(0,2,k)
#
#     std = 0.15
#     lb = (0-center)/std
#     ub = (1-center)/std
#     c_levels = truncnorm.rvs(
#         lb, ub, loc=center, scale=std, size=n)
#     dist_string = "N({},{})".format(center, std)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#     center = 0.55
#     difficulty = None  # np.random.normal(0,2,k)
#
#     std = 0.1
#     lb = (0-center)/std
#     ub = (1-center)/std
#     c_levels = truncnorm.rvs(
#         lb, ub, loc=center, scale=std, size=n)
#     dist_string = "N({},{})".format(center, std)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#     std = 0.2
#     lb = (0-center)/std
#     ub = (1-center)/std
#     c_levels = truncnorm.rvs(
#         lb, ub, loc=center, scale=std, size=n)
#     dist_string = "N({},{})".format(center, std)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#     std = 0.3
#     lb = (0-center)/std
#     ub = (1-center)/std
#     c_levels = truncnorm.rvs(
#         lb, ub, loc=center, scale=std, size=n)
#     dist_string = "N({},{})".format(center, std)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#
#     #-------------------------------------------------------------------------#
# ## IRT - Normal
#     lb = (0 - center) / std
#     ub = (1 - center) / std
#     c_levels = truncnorm.rvs(
#     lb, ub, loc=center, scale=std, size=n)
#     difficulty = np.random.normal(0, 1, k)
#     dist_string = "N({},{})_IRT_N({},{})".format(0, 1, 0, 1)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#
#     difficulty = np.random.normal(0, 2, k)
#     dist_string = "N({},{})_IRT_N({},{})".format(0, 1, 0, 2)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

## Good-Bad, with a fraction of good (ub) workers and all others bad (lb)
    good_fraction = 0.2
    lb, ub = 0.5, 0.8   # alpha = 0.8*0.2 + 0.5 * 0.8 = 0.56
    X = np.random.uniform(0, 1, n)
    c_levels = np.array([ub if (x < good_fraction) else lb for x in X])

    difficulty = None
    dist_string = "HS({},{},g={})".format(lb, ub, good_fraction)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty,balance=True)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
# # -------------------------------------------------------------------------#
# # ## IRT - Good/Bad
# #good_fraction = 0.2
#     lb, ub = -4, 1
#     X = np.random.uniform(0, 1, n)
#     c_levels = np.array([ub if (x < good_fraction) else lb for x in X])
#     difficulty = np.random.normal(0, 1, k)
#     dist_string = "GB({},{},g={})_IRT_N({},{})".format(lb, ub,good_fraction, 0, 1)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)
#
#     difficulty = np.random.normal(0, 2, k)
#     dist_string = "GB({},{},g={})_IRT_N({},{})".format(lb, ub,good_fraction, 0, 2)
#     s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
#     write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

def main_cat_more():
    possible_outcomes = 4
    k = 1000  # questions
    n = 1000  # workers
    write_to_dir = Path.cwd() / 'data' / 'synthetic'


    center = 0.3
    difficulty = None  # np.random.normal(0,2,k)
    std = 0.1
    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    dist_string = "N({},{})".format(center, std)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    center = 0.4
    difficulty = None  # np.random.normal(0,2,k)
    std = 0.1
    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    dist_string = "N({},{})".format(center, std)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    center = 0.5
    difficulty = None  # np.random.normal(0,2,k)
    std = 0.1
    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    dist_string = "N({},{})".format(center, std)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    center = 0.4
    difficulty = None  # np.random.normal(0,2,k)
    std = 0.3
    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    dist_string = "N({},{})".format(center, std)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    lb, ub = 0.2, 0.5

    # ---------------------------------------------------------- #
    # ## uniform in range [lb,ub]
    # p = np.random.uniform(lb, ub, n)  # competence
    # dist_name = "U"
    c_levels = np.random.uniform(lb, ub, n)  # competence
    difficulty = None  # np.random.normal(0,2,k)
    dist_string = "U({},{})".format(lb, ub)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    # ---------------------------------------------------------- #
    # triangular with mode = chance guess
    center = 1 / possible_outcomes
    c_levels = np.random.triangular(lb, center, ub, n)  # competence
    difficulty = None  # np.random.normal(0,2,k)
    dist_string = "T({},{},{})".format(lb, center, ub)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    ## Good-Bad, with a fraction of good (ub) workers and all others bad (lb)
    good_fraction = 0.2
    lb, ub = 0.24, 0.5  # alpha = 0.24 * 0.8 + 0.5 * 0.2 ~= 0.35
    X = np.random.uniform(0, 1, n)
    c_levels = np.array([ub if (x < good_fraction) else lb for x in X])

    difficulty = None
    dist_string = "GB({},{},g={})".format(lb, ub, good_fraction)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    ## IRT - Normal
    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    difficulty = np.random.normal(0, 1, k)
    dist_string = "N({},{})_IRT_N({},{})".format(0, 1, 0, 1)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    possible_outcomes = 8
    center = 0.3
    difficulty = None  # np.random.normal(0,2,k)
    std = 0.1

    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    dist_string = "N({},{})".format(center, std)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)

    possible_outcomes = 12
    center = 0.3
    difficulty = None  # np.random.normal(0,2,k)
    std = 0.1
    lb = (0-center)/std
    ub = (1-center)/std
    c_levels = truncnorm.rvs(
        lb, ub, loc=center, scale=std, size=n)
    dist_string = "N({},{})".format(center, std)
    s, gt, base_name = create_categorial_iid(n, k, possible_outcomes, c_levels, dist_string, difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, c_levels)


def main_cont():
    write_to_dir = Path.cwd()  / 'synthetic'
    m = 2000  # questions
    n = 2000  # workers

    # f_levels distribution:
    F_center = 1
    F_std = 0.5
    lb = (0-F_center)/F_std
    ub = (10-F_center)/F_std
    f_levels = truncnorm.rvs(
        lb, ub, loc=F_center, scale=F_std, size=n)

    # GT distribution:
    GT_center = 0
    GT_std = 2
    gt = np.random.normal(GT_center, GT_std, m)
    difficulty = None  # np.random.uniform(0.2, 5, k)
    dist_string = "GT=N({},{})_F=N({},{})".format(GT_center, GT_std, F_center,F_std)
    s, base_name = create_continuous_iid(n, m, f_levels, dist_string, gt,difficulty)
    write_data_to_file(n, s, gt, base_name, write_to_dir, f_levels)


def get_all_ranking_distance(k, gt):
    from itertools import permutations
    from proxytools.distances import kendall_tau_distance
    A = range(k)
    L = permutations(A)
    all_perms_at_distance_d = [[]]
    for j in range(int(k * (k - 1) / 2)):
        all_perms_at_distance_d.append([])
    for i, vec in enumerate(L):
        # dist_from_truth = kendall_tau_distance(vec, gt)
        dist_from_truth = kendall_tau_distance_k(vec, gt)

        all_perms_at_distance_d[dist_from_truth].append(vec)
    return all_perms_at_distance_d


def sample_mallows(phi, k, all_perms_at_distance_d):
    all_d = range(int(k * (k - 1) / 2 + 1))
    p_d = np.zeros(len(all_d))
    for d in all_d:
        p_d[d] = (phi ** d) * len(all_perms_at_distance_d[d])
    p_d = p_d / np.sum(p_d)
    # sample distance
    d_i = np.random.choice(all_d, p=p_d)
    all_relevant_perms = all_perms_at_distance_d[d_i]
    # sample permutation at distance
    select_set = range(len(all_relevant_perms))
    ind_i = np.random.choice(select_set)
    perm_i = all_relevant_perms[ind_i]
    return perm_i


# Creates the Mallows model distribution
def get_mallows_distribution(phi, k, gt):
    from itertools import permutations
    from proxytools.distances import kendall_tau_distance
    A = range(k)
    L = permutations(A)
    all_rankings = []
    p = np.zeros(math.factorial(k))
    dist_from_truth = np.zeros(math.factorial(k))
    for i, vec in enumerate(L):
        dist_from_truth[i] = kendall_tau_distance(vec, gt)
        p[i] = phi ** (distance.hamming(vec, gt, w=None) * k)
        all_rankings.append(vec)
    z = sum(p)
    p = list(p / z)
    data = {'ranking': all_rankings, 'p': p}
    ans = pd.DataFrame(data=data)
    return ans


def create_ranking_mallows(n, k, f_levels, dist_string):
    gt = range(k)
    base_name = 'Mallows_k{}_n{}_{}'.format(k, n, dist_string)
    # s = np.random.choice(ranking, size=n, replace=True, p=p)
    all_perms_at_distance_d = get_all_ranking_distance(k, gt)
    s = []
    for i in range(n):
        phi_i = f_levels[i]
        perm_i = sample_mallows(phi_i, k, all_perms_at_distance_d)
        s.append(perm_i)
    return s, gt, base_name


def test():
    possible_outcomes = 2
    k = 2000  # questions
    n = 2000  # workers
    lb, ub = 0.4, 0.8
    write_to_dir = Path.cwd() /  'synthetic'


def main_ranking():
    k = 4
    n = 20
    write_to_dir = Path.cwd() /  'Rankings' / 'Mallows'

    # # dist_name = "U"
    # lb = 0.3
    # ub = 1
    # f_levels = np.random.uniform(lb, ub, n)  # competence
    # dist_string = "U({},{})".format(lb, ub)
    #
    # s, gt, base_name = create_ranking_mallows(n, k,  f_levels, dist_string)
    # write_ranking_data_to_file(n, s, gt, base_name, write_to_dir, levels=None)

    center = 0.65
    std = 0.15
    f_levels = np.random.normal(center, std, n)
    f_levels = bound(0, 1, f_levels)
    dist_string = "N({},{})".format(center, std)

    s, gt, base_name = create_ranking_mallows(n, k, f_levels, dist_string)
    write_ranking_data_to_file(n, s, gt, base_name, write_to_dir, levels=None)

    center = 0.85
    std = 0.15
    f_levels = np.random.normal(center, std, n)
    f_levels = bound(0, 1, f_levels)
    dist_string = "N({},{})".format(center, std)

    s, gt, base_name = create_ranking_mallows(n, k, f_levels, dist_string)
    write_ranking_data_to_file(n, s, gt, base_name, write_to_dir, levels=None)

    center = 0.85
    std = 0.25
    f_levels = np.random.normal(center, std, n)
    f_levels = bound(0, 1, f_levels)
    dist_string = "N({},{})".format(center, std)

    s, gt, base_name = create_ranking_mallows(n, k, f_levels, dist_string)
    write_ranking_data_to_file(n, s, gt, base_name, write_to_dir, levels=None)

    ## Good-Bad, with a fraction of good (ub) workers and all others bad (lb)
    good_fraction = 0.2
    lb, ub = 0.3, 0.99  # alpha = 0.8*0.2 + 0.48 * 0.8 ~= 0.55
    X = np.random.uniform(0, 1, n)
    f_levels = np.array([lb if (x < good_fraction) else ub for x in X])

    dist_string = "GB({},{},g={})".format(lb, ub, good_fraction)
    s, gt, base_name = create_ranking_mallows(n, k, f_levels, dist_string)
    write_ranking_data_to_file(n, s, gt, base_name, write_to_dir, levels=None)


if __name__ == '__main__':

    # main_cat_binary()
    # main_cat_more()
    #main_ranking()
    main_cont()
    #test()

