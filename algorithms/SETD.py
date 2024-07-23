import pickle
import pystan

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from itertools import combinations
from functools import partial
from dataclasses import dataclass, asdict, field
from utils.helpers import get_pairwise_distances, get_pairwise_distances_ranking, compute_weight_from_fault
from utils.aggregations import aggregate


"""
This code ONLY RUNS ON A LINUX machine as pystan is not fully supported for windows yet. 

Paper: Braylan, Lease, "Modeling Complex Annotations"
Link: https://www.ischool.utexas.edu/~ml/papers/braylan-hcompdc19.pdf
@author: Alex Braylan
Source code: https://github.com/Praznat/annotationmodeling

INPUT: Any distance-based data
"""


class Const:
    stan_model_path = Path(
        "/home//tsviel//PycharmProjects//proxy_crowdsourcing//proxytools//proxy_crowsourcing//stan_models//")


def user_avg_dist(stan_data, apply_empirical_prior=True):
    """ BAU scores for each user = user's average distance across whole dataset """
    sddf = pd.DataFrame(stan_data)
    s1 = sddf.groupby("u1s").sum()["distances"]
    s2 = sddf.groupby("u2s").sum()["distances"]
    n1 = sddf.groupby("u1s").count()["distances"]
    n2 = sddf.groupby("u2s").count()["distances"]
    count = n1.add(n2, fill_value=0)
    avg_distances = s1.add(s2, fill_value=0) / count

    if apply_empirical_prior:
        prior_mu, prior_var = avg_distances.mean(), avg_distances.var()
        sddf["u1delta"] = np.square(sddf["distances"] - avg_distances[sddf["u1s"]].values)
        sddf["u2delta"] = np.square(sddf["distances"] - avg_distances[sddf["u2s"]].values)
        v1 = sddf.groupby("u1s").sum()["u1delta"]
        v2 = sddf.groupby("u2s").sum()["u2delta"]
        var_distances = v1.add(v2, fill_value=0) / count
        var_distances = var_distances.fillna(0.01)
        var_distances += 0.1 * prior_var  # to avoid division by zero variance
        nominator = prior_mu / prior_var + count * avg_distances / var_distances
        denominator = 1 / prior_var + count / var_distances
        avg_distances = nominator / denominator
    assert not np.isnan(avg_distances.values.sum())
    return avg_distances


def user_nearest_gold(stan_data):
    sddf = pd.DataFrame(stan_data)
    scores = sddf[sddf["u1s"] == 1].groupby("u2s").mean()["distances"]
    score_mu = scores[scores > scores.min()].mean()
    return scores.reindex(range(1, stan_data["NUSERS"] + 1), fill_value=score_mu)


def get_model_user_rankings(opt):
    """ MAS model's user annotation ranking by item """
    errs = opt["dist_from_truth"]
    result = np.argsort(errs, axis=1)
    tmp = errs[0][result[0]]
    assert tmp[0] <= tmp[1]
    return result


def stan_model(model_name):
    pickle_file = Const.stan_model_path.joinpath(f'{model_name}.pkl')
    if pickle_file.exists():
        stan_model_file = pickle.load(open(pickle_file, 'rb'))
    else:
        stan_model_file = pystan.StanModel(file=Const.stan_model_path.joinpath(f'{model_name}.stan').open())
        print("Pickling model")
        with open(pickle_file, 'wb') as f:
            pickle.dump(stan_model_file, f)
    return stan_model_file


# def preds_from_opt(opt, matrix):
#     # per_item_user_rankings = get_model_user_rankings(opt)
#     # best_annotations = per_item_user_rankings[:, 0]
#     # answers = [matrix[user, item] for item, user in enumerate(per_item_user_rankings)]
#     estimated_fault = opt['uerr']
#     return best_annotations, answers


@dataclass
class StanData:
    distance_matrix: np.array

    u1s: np.array = field(default_factory=list)
    u2s: np.array = field(default_factory=list)
    distances: np.array = field(default_factory=list)
    items: np.array = field(default_factory=list)

    n_gold_users: int = 0
    gold_user_err: int = 0
    use_norm: int = 1
    eps_limit: int = 3
    uerr_prior_loc_scale: int = 8
    diff_prior_loc_scale: int = 8

    diff_prior_scale: float = .0251
    err_scale: float = .1
    uerr_prior_scale: float = .251
    use_uerr: float = 1
    use_diff: float = 1
    norm_ratio: float = 1

    DIM_SIZE: int = 8
    NDATA: int = 0
    NUSERS: int = 0
    NITEMS: int = 1

    def __post_init__(self):
        user_pairs = list(combinations(range(self.distance_matrix.shape[0]), 2))
        for u1, u2 in user_pairs:
            self.u1s.append(u1)
            self.u2s.append(u2)
            self.distances.append(self.distance_matrix[u1, u2])

        self.u1s = np.array(self.u1s) + 1
        self.u2s = np.array(self.u2s) + 1
        self.items = np.array([1] * len(self.u1s))
        self.distances = np.array(self.distances) + (.1 - np.min(self.distances))

        self.NDATA = len(self.distances)
        self.NUSERS = self.distance_matrix.shape[0]

    def asdict(self):
        return {k: v for k, v in self.__dict__.items() if v is not self.distance_matrix}


def re_shaper(item: np.array, distance_function):
    return distance_function(item.reshape(-1, 1))


def collect_distances(matrix, user_pairs):
    distances = []
    u1s = []
    u2s = []

    for i, j in user_pairs:
        distance = matrix[i, j]
        if np.isnan(distance):
            continue
        distances.append(distance)
        u1s.append(i)
        u2s.append(j)

    return distances, u1s, u2s


def generate_stan_data(matrix, distance_function):
    user_pairs = list(combinations(range(matrix.shape[0]), 2))
    u1_ids = []
    u2_ids = []
    for u1, u2 in user_pairs:
        u1_ids.append(u1)
        u2_ids.append(u2)

    metric = partial(re_shaper, distance_function=distance_function)
    item_matrices = np.apply_along_axis(metric, 0, matrix)
    distance_collector = partial(collect_distances, user_pairs=user_pairs)

    u1s = []
    u2s = []
    items = []
    distances = []

    for i in tqdm(range(item_matrices.shape[-1])):
        dis, u1, u2 = distance_collector(item_matrices[:, :, i])
        distances += dis
        u1s += u1
        u2s += u2
        items += [i] * len(u1)

    stan_data = StanData(items=np.array(items),
                         u1s=np.array(u1s),
                         u2s=np.array(u2s),
                         distances=np.array(distances),
                         NUSERS=matrix.shape[0]
                         )
    return asdict(stan_data)


def SETD(data: pd.DataFrame,
         gt,
         pariwise_distance_function,
         voting_rule,
         params,
         mode="dictator",
         skip_stan=False,
         weight_transform="simple",
         iterations=1500,
         positive=True,
         normalize=True):
    # stan_data = generate_stan_data(dist_matrix, distance_function)

    if mode == "rankings":
        possible_answers = 2
        dist_matrix = get_pairwise_distances_ranking(data, gt, pariwise_distance_function)
    else:
        possible_answers = set(np.unique(data.values))
        dist_matrix = get_pairwise_distances(data, pariwise_distance_function)

    if skip_stan:  # this is to test the function when outside virtual machine
        estimated_fault = np.ones(data.shape[0]) / 4
        print("Skip STAN - test only")
    else:
        stan_data = StanData(dist_matrix).asdict()
        uerr_b = user_avg_dist(stan_data).values

        init = {
            "uerr_Z": uerr_b - np.mean(uerr_b),
            "uerr": uerr_b,
            "diff": np.ones(stan_data["NITEMS"])
        } if stan_data["use_uerr"] else {}

        stan_enriched_data = dict(**stan_data, gold_uerr=user_nearest_gold(stan_data))
        mas_model = stan_model("mas2")
        mas_iter = iterations
        mas_opt = mas_model.optimizing(data=stan_enriched_data, init=init, verbose=False, iter=mas_iter)

        estimated_fault = mas_opt['uerr'] / mas_opt['uerr'].sum()

    weights = compute_weight_from_fault(estimated_fault, mode, positive, normalize, weight_transform)
    answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)
    return answers