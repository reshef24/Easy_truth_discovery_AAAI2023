# -*- coding: utf-8 -*-
"""
Wan, Chen et. al. "From Truth Discovery to Trustworthy Opinion Discovery:
An Uncertainty-Aware Quantitative Modeling Approach"
Link: http://hanj.cs.illinois.edu/pdf/kdd16_mwan.pdf

KDEm.py
@author: Mengting Wan
SourceCode: https://github.com/MengtingWan/KDEm

Some runs fail due to division by zero. These are excluded from analysis.

KDEm_alg is a wrapper function that interfaces with the original code.
INPUT: Real valued data
PARAMS:
    - data_params: a dictionary containing the data and metadata
    - params: a dictionary containing the parameters for the algorithm
"""

import numpy as np
import numpy.linalg as la
from utils.helpers import compile_param_list

tol = 1e-5


# update source reliability scores
def update_c(index, m, n, count, norm_M, method):
    rtn = np.zeros(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + norm_M[i] / len(index[i])
    tmp = np.sum(rtn)
    if tmp > 0:
        rtn[rtn > 0] = np.copy(-np.log((rtn[rtn > 0] / count[rtn > 0]) / tmp))
    return [rtn, tmp]


# update opinion distributions
def update_w(index, m, n, c_vec, norm_M, method):
    w_M = []
    for i in range(n):
        w_i = np.zeros(len(index[i]))
        tmp = c_vec[index[i]]
        w_i[norm_M[i] > 0] = tmp[norm_M[i] > 0]
        tmp1 = sum(w_i)
        if tmp1 > 0:
            w_M.append(w_i / tmp1)
        else:
            w_i[norm_M[i] == 0] = 1
            tmp1 = sum(w_i)
            w_M.append(w_i / tmp1)
    return w_M

"""
A wrapper function that interfaces with the original code
"""

def KDEm_alg(data_params, params):
    argmax = params.get("argmax", False)
    m, n = data_params['df'].shape
    kernel = params["kernel"]
    outlier_thr = params.get("outlier_thr", 0)
    h = params.get("h", -1)
    norm = params.get("norm", True)
    max_itr = params.get("iterations", 100)
    data_as_rows = [list(i) for i in dict(data_params["df"]).values()]

    # translate the data to the format used by the Wan et al. code :
    data_raw = []
    for worker in data_as_rows:
        w = []
        for idx, answer in enumerate(worker):
            w.append([idx, answer, 1])
        data_raw.append(np.array(w))

    if norm:
        data, data_mean, data_sd = normalize(data_raw)
    else:
        data = data_raw[:]

    source_score, weights_for_each, itr = KDEm(data, m, n, max_itr=max_itr, method=kernel, h=h)
    out, cluster_index, cluster_confidence = wKDE_twist(data, m, n, weights_for_each, kernel, argmax, outlier_thr, h)
    # moments = get_moments(data, m, n, weights_for_each, method=kernel, h=h)
    if norm:
        truth_out = normalize_ivr(out, data_mean, data_sd)
    else:
        truth_out = out[:]

    # the original code takes the best answer for each question
    answers = refine_single(truth_out, cluster_confidence)
    return answers


# implement KDEm without claim-value mappings

def KDEm(data, m, n, tol=1e-5, max_itr=99, method="Gaussian", h=-1):
    err = 99
    index, claim, count = extract(data, m, n)
    w_M = []
    for i in range(n):
        l = len(index[i])
        w_M.append(np.ones(l) / l)
    itr = 1
    kernel_M = get_kernel_matrix(claim, n, method)
    norm_M = get_norm_matrix(kernel_M, n, w_M, method)
    c_vec, J = update_c(index, m, n, count, norm_M, method)
    while (err > tol) & (itr < max_itr):
        itr = itr + 1
        J_old = J
        c_old = np.copy(c_vec)
        w_M = update_w(index, m, n, c_old, norm_M, method)
        norm_M = get_norm_matrix(kernel_M, n, w_M, method)
        c_vec, J = update_c(index, m, n, count, norm_M, method)
        # err = la.norm(c_vec - c_old)/la.norm(c_old)
        err = abs((J - J_old) / J_old)
    return [c_vec, w_M, itr]


def get_kernel_matrix(claim, n, method, h=-1):
    kernel_M = []
    for i in range(n):
        x_i = claim[i]
        if h < 0:
            h = MAD(x_i)
            # h = np.std(x_i)
        l = x_i.shape[0]
        tmp = np.zeros((l, l))
        for j in range(l):
            if h > 0:
                tmp[j, :] = K((x_i[j] - x_i) / h, method)
            else:
                tmp[j, :] = K(0, method)
        kernel_M.append(tmp)
    return kernel_M


def get_norm_matrix(kernel_M, n, w_M, method):
    norm_M = []
    for i in range(n):
        kernel_m = kernel_M[i]
        term1 = np.diag(kernel_m)
        term2 = np.dot(kernel_m, w_M[i])
        term3 = np.dot(w_M[i], term2)
        tmp = term1 - 2 * term2 + term3
        tmp[tmp < 0] = 0
        norm_M.append(tmp)
    return norm_M


def K(x, method="Gaussian"):
    rtn = 0
    if method.lower() == "uniform":
        rtn = (abs(x) <= 1) / 2
    if method.lower() == "epanechnikov" or method.lower() == "ep":
        rtn = 3 / 4 * (1 - x ** 2) * (abs(x) <= 1)
    if method.lower() == "biweight" or method.lower() == "bi":
        rtn = 15 / 16 * (1 - x ** 2) ** 2 * (abs(x) <= 1)
    if method.lower() == "triweight" or method.lower() == "tri":
        rtn = 35 / 32 * (1 - x ** 2) ** 3 * (abs(x) <= 1)
    if method.lower() == "gaussian":
        rtn = np.exp(-x ** 2) / np.sqrt(2 * np.pi)
    if method.lower() == "laplace":
        rtn = np.exp(-abs(x))
    return rtn


def MAD(x_i):
    return np.median(abs(x_i - np.median(x_i, axis=0)), axis=0) + 1e-10 * np.std(x_i, axis=0)


def extract(data, m, n):
    index = []
    claim = []
    count = np.zeros(m)
    for i in range(n):
        src = list(data[i][:, 0].astype(int))
        count[src] = count[src] + 1
        index.append(src)
        claim.append(data[i][:, 1])
    return [index, claim, count]


def get_density(t, x_i, w_i, h):
    if h > 0:
        tmp = np.dot(w_i, K((t - x_i) / h)) / (h * sum(w_i))
    else:
        tmp = 1
    return tmp


def DENCLUE(x_i, wi_vec, method="gaussian", tol=1e-8, h=-1):
    def cluster_update(x_old, x_i, wi_vec, h, method):
        l = len(wi_vec)
        tmp0 = np.ones((l, l))
        tmp1 = np.ones((l, l))
        if method.lower() == "epanechnikov" or "ep" == method.lower():
            for i in range(l):
                tmp0[:, i] = K((x_old[i] - x_i) / h, method="uniform")
                tmp1[:, i] = tmp0[:, i] * x_i
        if method.lower() == "biweight" or method.lower() == "bi":
            for i in range(l):
                tmp0[:, i] = K((x_old[i] - x_i) / h, method="ep")
                tmp1[:, i] = tmp0[:, i] * x_i
        if "triweight" == method.lower() or "tri" == method.lower():
            for i in range(l):
                tmp0[:, i] = K((x_old[i] - x_i) / h, method="bi")
                tmp1[:, i] = tmp0[:, i] * x_i
        if method.lower() == "gaussian":
            for i in range(l):
                tmp0[:, i] = K((x_old[i] - x_i) / h)
                tmp1[:, i] = tmp0[:, i] * x_i
        rtn = np.dot(wi_vec, tmp1)
        rtn_de = np.dot(wi_vec, tmp0)
        rtn[rtn_de > 0] = rtn[rtn_de > 0] / rtn_de[rtn_de > 0]
        return rtn

    err = 99
    if h < 0:
        h = MAD(x_i)
        # h = np.std(x_i)
    if sum(wi_vec) == 0:
        wi_vec = wi_vec + 1e-5
    if np.var(x_i) > 0:
        x_new = np.copy(x_i) + 1e-12
        while err > tol:
            x_old = np.copy(x_new)
            x_new = cluster_update(x_old, x_i, wi_vec, h=h, method=method)
            err = la.norm(x_old - x_new) / la.norm(x_old)
    else:
        x_new = np.copy(x_i)
    return x_new


def twist(x, x_i, wi_vec, argmax=False, cut=0, h=-1, tol=1e-3):
    l = len(x)
    center = np.array([x[0]])
    if (h < 0):
        h = MAD(x_i)
        # h = np.std(x_i)
    conf = np.array([get_density(x[0], x_i, wi_vec, h)])
    ind = np.zeros(l)
    if (sum(wi_vec) == 0):
        wi_vec = wi_vec + 1e-5
    for i in range(1, l):
        if 0 in center:
            print("here")
        tmp = abs((x[i] - center) / center)
        if (tmp.min() > tol):
            center = np.append(center, x[i])
            conf = np.append(conf, get_density(x[i], x_i, wi_vec, h))
            ind[i] = len(center) - 1
        else:
            ind[i] = tmp.argmin()
    conf = conf / sum(conf)
    if (cut > 0):
        tmp = np.where(conf > cut)[0]
        center_new = center[list(tmp)]
        ind_new = -np.ones(l)
        for i in range(len(center_new)):
            ind_new[ind == tmp[i]] = i
        center = np.copy(center_new)
        ind = np.copy(ind_new)
        conf = conf[conf > cut]
        conf = conf / sum(conf)

    if argmax:
        for i in range(len(center)):
            tmp0 = x_i[ind == i]
            tmp1 = np.zeros(len(tmp0))
            for j in range(len(tmp0)):
                tmp1[j] = get_density(tmp0[j], x_i, wi_vec, h)
            center[i] = tmp0[np.argmax(tmp1)]
    return [center, ind, conf]


def wKDE_twist(data, m, n, w_M, method, argmax, cut=0, h=-1):
    index, claim, count = extract(data, m, n)
    n = len(claim)
    truth = []
    ind_c = []
    conf = []
    for i in range(n):
        x_new = DENCLUE(claim[i], w_M[i], method, h=h)
        center, ind, conf_i = twist(x_new, claim[i], w_M[i], argmax, cut, h=h)
        truth.append(center)
        ind_c.append(ind)
        conf.append(conf_i)
    return [truth, ind_c, conf]


def normalize(data):
    data_new = []
    n = len(data)
    data_mean = np.zeros(n)
    data_sd = np.zeros(n)
    for i in range(n):
        data_i = np.float64(np.copy(data[i]))
        data_mean[i] = np.mean(data[i][:, 1])
        data_sd[i] = np.std(data[i][:, 1])
        if data_sd[i] > 0:
            data_i[:, 1] = (data[i][:, 1] - data_mean[i]) / data_sd[i]
        else:
            data_i[:, 1] = (data[i][:, 1] - data_mean[i])
        data_new.append(data_i)
    return [data_new, data_mean, data_sd]


def normalize_ivr(data, data_mean, data_sd):
    data_new = []
    n = len(data)
    for i in range(n):
        data_i = np.copy(data[i]) * data_sd[i] + data_mean[i]
        data_new.append(data_i)
    return data_new


def get_moments(data, m, n, w_M, method="gaussian", h=-1):
    moments = np.zeros((n, 3))
    for i in range(n):
        x_i = np.copy(data[i][:, 1])
        if len(w_M) > 0:
            moments[i, :] = bsf_get_moments(x_i, w_M[i], h)
        else:
            moments[i, :] = bsf_get_moments(x_i, np.ones(len(x_i)) / len(x_i), h)
    return moments


def bsf_get_moments(x_i, wi_vec, h, method="gaussian"):
    if (h < 0):
        h = MAD(x_i)
        # h = np.std(x_i)
    mu = np.dot(wi_vec, x_i)
    if (method == "laplace"):
        m2 = np.dot(wi_vec, (x_i - mu) ** 2 + 2 * h ** 2)
        m3 = np.dot(wi_vec, (x_i - mu) ** 3)
        m4 = np.dot(wi_vec, (x_i - mu) ** 4 + 6 * (x_i - mu) ** 2 * 2 * h ** 2 + 24 * h ** 4)
    else:
        m2 = np.dot(wi_vec, (x_i - mu) ** 2 + h ** 2)
        m3 = np.dot(wi_vec, (x_i - mu) ** 3)
        m4 = np.dot(wi_vec, (x_i - mu) ** 4 + 6 * (x_i - mu) ** 2 * h ** 2 + 3 * h ** 4)
    skewness = m3 / (m2) ** (3 / 2)
    kurtosis = m4 / (m2) ** 2 - 3
    stats = skewness * 2 - kurtosis
    return [skewness, kurtosis, stats]


def refine_single(truth, conf):
    truth_single = []
    n = len(truth)
    for i in range(n):
        truth_tmp = np.sort(truth[i])
        conf_tmp = conf[i][np.argsort(truth[i])]
        if len(truth_tmp) > 0:
            truth_single.append(truth_tmp[np.argmax(conf_tmp)])
        else:
            truth_single.append(np.array([]))
    return np.array(truth_single)