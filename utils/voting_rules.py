import numpy as np
import pandas as pd

from scipy import special
from itertools import permutations
from measures.vector_distance_functions import kendall_tau_distance
from measures.pairwise_distance_functions import hamming_binary_pairwise
from utils.helpers import w_random, str_to_coord_list, coord_row_to_str


def plurality_internal(A, possible_answers_num=2, weights=None, exclude_agents=None, signed=False):
    if isinstance(A, pd.DataFrame):
        A = A.values
    n = A.shape[0]
    if weights is None:
        weights = np.ones(n) / n

    if len(A.shape) > 1:
        if signed:
            k = A.shape[0]
        else:
            k = A.shape[1]
    else:
        k = 1
    if possible_answers_num == 2:
        if not signed:
            A = (A * 2 - 1).T
        sums = weights * A
        x = np.nansum(sums.T, axis=0)  # required for sparse data
        y = np.random.uniform(-0.00001, 0.00001, k)
        answers = np.sign(x + y)
        if not signed:
            answers = (answers + 1) / 2
    else:
        counts = np.zeros((possible_answers_num, k))
        for z in range(possible_answers_num):
            Az = weights * (A == z).T
            counts[z, :] = np.sum(Az.T, axis=0) + np.random.uniform(0, 0.000001, k)
        answers = np.argmax(counts, axis=0)
    return answers


def vr_random_dictator(s, w=None, **kwargs):
    if isinstance(s, list):
        s = pd.DataFrame(s)  # convert List of Lists to DataFrame
    dictator = w_random(s.shape[0], w)
    return list(s.loc[dictator, :])


def vr_dictatorship_of_fittest(s, w=None, **kwargs):
    if isinstance(s, list):
        s = pd.DataFrame(s)  # convert List of Lists to DataFrame
    n = s.shape[0]
    if w is None:
        w = np.ones(n) / n
    tie_break = np.random.uniform(0, 0.00001, len(w))
    w = w + 0.001 * tie_break
    i = np.argmax(w)
    return list(s.loc[i])


def vr_plurality(rankings, w=None, **kwargs):
    alternatives = np.sort(rankings[0])
    if isinstance(rankings, list):
        rankings = pd.DataFrame(rankings)  # convert List of Lists to DataFrame
    n = rankings.shape[0]
    k = rankings.shape[1]
    max_k = np.max(alternatives)
    if w is None:
        w = np.ones(n) / n
    scores = np.histogram(a=rankings[0], bins=range(max_k + 3), weights=w)[0]
    for a in alternatives:
        scores[a] += max_k
    tie_break = np.random.uniform(0, 0.00001, len(scores))
    scores += tie_break
    result = np.argsort(-scores)[0:k]
    return list(result)


def vr_veto(rankings, w=None, **kwargs):
    alternatives = np.sort(rankings[0])
    if isinstance(rankings, list):
        rankings = pd.DataFrame(rankings)  # convert List of Lists to DataFrame
    n = rankings.shape[0]
    k = rankings.shape[1]
    max_k = np.max(alternatives)
    if w is None:
        w = np.ones(n) / n
    scores = np.histogram(a=rankings[k - 1], bins=range(max_k + 3), weights=w)[0]
    for a in alternatives:
        scores[a] -= max_k
    tie_break = np.random.uniform(0, 0.00001, len(scores))
    scores += tie_break
    result = np.argsort(scores)[0:k]
    return list(result)


def vr_WKY_graph(df, all_ranking_as_pairwise, candidates=None, w=None, **kwargs):
    n = df.shape[0]
    if w is None:
        w = np.ones(n) / n
    aggregated = np.dot(w, df.values)
    tie_break = np.random.uniform(0, 0.00000001, len(all_ranking_as_pairwise))
    distances = np.abs(aggregated - all_ranking_as_pairwise).sum(axis=1) + tie_break
    return "index", np.argmin(distances)


def vr_KY_graph(df, all_ranking_as_pairwise, candidates=None, w=None, **kwargs):
    n = df.shape[0]
    if w is None:
        w = np.ones(n) / n
    aggregated = np.dot(w, df.values)
    aggregated = aggregated > 0.5 + np.random.uniform(-0.000001, 0.000001, aggregated.shape)
    tie_break = np.random.uniform(0, 0.00000001, len(all_ranking_as_pairwise))
    distances = np.abs(aggregated - all_ranking_as_pairwise).sum(axis=1) + tie_break
    return "index", np.argmin(distances)


def pairwise_to_scores(pairwise, candidates):
    j_inx = 0
    k = candidates
    scores = np.zeros(k)
    for i in range(k):
        for j in range(i + 1, k):
            scores[i] += pairwise[j_inx]
            scores[j] += (1 - pairwise[j_inx])
            j_inx = j_inx + 1
    return scores


def vr_borda_graph(df, all_ranking_as_pairwise, alternatives, w=None, **kwargs):
    n = df.shape[0]
    k = len(alternatives)
    if w is None:
        w = np.ones(n) / n
    aggregated = np.dot(w, df.values)
    scores = pairwise_to_scores(aggregated, k)
    tie_break = np.random.uniform(0, 0.0000001, k)
    scores += tie_break
    result_indices = np.argsort(-scores)
    return "list", [alternatives[x] for x in result_indices]


def vr_copeland_graph(df, all_ranking_as_pairwise, alternatives, w=None, **kwargs):
    n = df.shape[0]
    k = len(alternatives)
    if w is None:
        w = np.ones(n) / n
    aggregated = np.dot(w, df.values) > 0.5
    scores = pairwise_to_scores(aggregated, k)
    tie_break = np.random.uniform(0, 0.0000001, k)
    scores += tie_break
    result_indices = np.argsort(-scores)
    return "list", [alternatives[x] for x in result_indices]


# writen by VLAD NICULAE
# from https://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
def vr_kemeny_young_brute(rankings, w=None, **kwargs):
    if isinstance(rankings, pd.DataFrame):
        rankings = rankings.to_numpy()
    if isinstance(rankings, list):
        rankings = np.array(rankings)
    n_voters, n_candidates = rankings.shape
    if w is not None:
        w[w < 0] = 0
    else:
        w = np.ones(n_voters)
    candidates = sorted(rankings[0])
    min_dist = np.inf
    best_rank = None
    for candidate_rank in permutations(candidates):
        dist = 0
        for i, voter_rank in enumerate(rankings):
            dist += kendall_tau_distance(candidate_rank, voter_rank) * w[i]
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return list(best_rank)


# implemented from https://www.sciencedirect.com/science/article/pii/S0004370215001587
def vr_procaccia(pi, L, alternatives, w=None, **kwargs):
    t_param = kwargs['t_param'] if kwargs.get('t_param') else 2
    t = maximal_distance(pi) / len(pi[0]) / t_param
    np_pi = np.array(pi)

    distance_L_pi = compute_hamming_distances_between_matrix_a_to_b(L, np_pi)
    if w is not None:
        distance_L_pi = np.multiply(distance_L_pi, w).sum(axis=1)
    else:
        distance_L_pi = distance_L_pi.mean(axis=1)

    B_t_k_indices = np.argwhere(distance_L_pi <= t)
    B_t_k = L[B_t_k_indices.reshape(len(B_t_k_indices), ), :]

    if not B_t_k.any():
        return vr_WKY_graph(pi, L, alternatives, w, **kwargs)
    distance_L_B_t_k = compute_hamming_distances_between_matrix_a_to_b(L, B_t_k)
    minimax_values = distance_L_B_t_k.max(axis=1)
    minimax_values = minimax_values + np.random.uniform(0, 0.00000001, len(minimax_values))
    minimax_order = minimax_values.argmin()
    return "index", minimax_order


def maximal_distance(voting_profile) -> float:
    """
    :param voting_profile:
    :return; the maximal distance between two voters in a voting profile
    """
    distances = hamming_binary_pairwise(voting_profile)
    maximal_dis = distances.max()
    return maximal_dis


def compute_hamming_distances_between_matrix_a_to_b(a, b):
    """
    This function will compute the distance between  every row in a to every row in b
    :param a:
    :param b:
    :return: matrix len(a) X len(b)
    """
    a_b = pd.DataFrame(np.concatenate([b, a]))
    distances = hamming_binary_pairwise(a_b) / len(a[0])
    distances = distances[len(b):, :len(b)]
    return distances


""" what does it do? """


def vr_modal_ranking(pi: pd.DataFrame, w=None, **kwargs):
    string_rankings = [str(list(row)) for row in pi]
    w = np.array([1] * len(string_rankings)) if w is None else w
    df = pd.DataFrame({"votes": string_rankings, "w": w}).groupby('votes').sum()
    return eval(df['w'].idxmax())


""" aggregates rankings using the VR function. needed for the KY rule which required more data params """


def ranking_to_pairwise_q(ranking_data, gt) :
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


def get_all_ranking_as_pairwise(gt):
    # Extract the inner list from ranking_gt
    ranking_gt = gt[0]
    all_ranking = permutations(ranking_gt)
    all_ranking = [list(map(int, x)) for x in list(all_ranking)]
    all_ranking_as_pairwise = ranking_to_pairwise_q(all_ranking, ranking_gt)
    return all_ranking, all_ranking_as_pairwise


def format_data_to_ranking_methods(df, gt):
    """Copied from original to save time"""
    ranking_data = df.to_numpy().tolist()
    ranking_gt = gt.to_numpy().tolist()
    all_ranking, all_ranking_as_pairwise = get_all_ranking_as_pairwise(ranking_gt)
    df = pd.DataFrame(ranking_to_pairwise_q(ranking_data, ranking_gt[0]))
    gt = [1] * (df.shape[1])

    data = {"ranking_df": ranking_data,
            "ranking_gt": ranking_gt,
            "df": df,
            "gt": gt,
            "all_ranking": all_ranking,
            "all_ranking_as_pairwise": all_ranking_as_pairwise}

    return data


def run_vr(data: pd.DataFrame, gt_ranking, vr_func, weights=None, **kwargs):
    data_params = format_data_to_ranking_methods(data, gt_ranking)

    vr_name = vr_func.__name__
    if weights is not None:
        weights = np.maximum(weights, 0)
    if vr_name in pairwise_vr_methods:
        candidates = np.sort(data_params["ranking_df"][0])
        answer_type, answers = vr_func(data, data_params["all_ranking_as_pairwise"], candidates, weights, **kwargs)
        if answer_type == "index":
            answers = data_params["all_ranking"][answers]

    else:
        answers = vr_func(data_params["ranking_df"], weights)
    return answers


# input is list of size-2-lists [x,y]
def coord_to_bitmap(coord, width, height, one_dim=False):
    if one_dim:
        bitmap = np.zeros(width * height)
    else:
        bitmap = np.zeros((width, height))
    for item in coord:
        x = item[0]
        y = item[1]
        if one_dim:
            bitmap[x * height + y] = 1
        else:
            bitmap[x, y] = 1
    return bitmap


# 1d
def bitmap_to_coord(bitmap, width, height, one_dim=False):
    coord = []
    for x in range(width):
        for y in range(height):
            if one_dim:
                pixel_on = bitmap[x * height + y] > 0
            else:
                pixel_on = bitmap[x, y] > 0
            if pixel_on:
                coord.append([x, y])
    return coord


def bitmap_majority(df, weights, **kwargs):
    n = len(df)
    # TODO: hard coded! bad!
    width = 201
    height = 201
    answer_matrix = np.zeros((n, width * height))
    for i in range(n):
        coord_str = df.iloc[i][0]
        coord_list = str_to_coord_list(coord_str, items_as_str=False)
        answer_matrix[i, :] = np.asarray(coord_to_bitmap(coord_list, width, height, one_dim=True))
    majority_bitmap = plurality_internal(answer_matrix, possible_answers_num=2, weights=weights, signed=False)
    majority_coord = bitmap_to_coord(majority_bitmap, width, height, one_dim=True)
    return str(majority_coord)


voting_rules = (vr_random_dictator,
                vr_dictatorship_of_fittest,
                vr_plurality,
                vr_veto,
                vr_WKY_graph,
                vr_KY_graph,
                pairwise_to_scores,
                vr_borda_graph,
                vr_copeland_graph,
                vr_kemeny_young_brute,
                vr_procaccia,
                vr_modal_ranking,
                bitmap_majority)

voting_rules_dict = {i.__name__: i for i in voting_rules}
pairwise_vr_methods = {"vr_KY_graph", "vr_WKY_graph", "vr_borda_graph", "vr_copeland_graph", "vr_procaccia"}
