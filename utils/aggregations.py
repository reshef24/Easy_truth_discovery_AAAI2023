import random
import numpy as np

from scipy.spatial import distance

from utils.voting_rules import voting_rules_dict, run_vr, plurality_internal
from utils.helpers import w_random
from utils.wquantile import wmedian


# weighted plurality of all workers
# signed means the matrix is over {-1,1} and transposed to k*n
def plurality(A, possible_answers_num=2, weights=None, exclude_agents=None, signed=False):
    return plurality_internal(A, possible_answers_num, weights, exclude_agents, signed)


def aggregate(data, gt, weights, voting_rule, mode, possible_answers, params):
    n, m = data.shape[0], data.shape[1]
    if weights is None:
        weights = np.ones(n) / n

    if voting_rule == "best_dictator":
        # select the worker with highest weight. Break ties randomly.
        dictator = random.choice(np.argwhere(weights == np.amax(weights)))[0]
        answers = np.array(data.iloc[dictator])

    elif voting_rule == "random_dictator":
        dictator = w_random(n, weights)
        answers = np.array(data.iloc[dictator])

    elif mode == 'continuous':
        if voting_rule == "mean":
            answers = np.average(data, axis=0, weights=weights, returned=False)
        elif voting_rule == "median":
            answers = np.zeros(m)
            for j in range(m):
                answers[j] = wmedian(data[j], weights)

    elif mode == "categorical":
        if isinstance(possible_answers, set):
            possible_answers = len(possible_answers)
        answers = plurality(data, possible_answers, weights)

    elif mode == "rankings":
        voting_rule = voting_rules_dict[voting_rule]
        if params is None: ## got error of params being none
            params = {}
        answers = run_vr(data, gt, voting_rule, weights, **params)

    elif mode == "etchacell":
        voting_rule = voting_rules_dict[voting_rule]
        answers = voting_rule(data, weights)
    else:
        raise Exception(f"Not Implemented")
    return answers


def compute_distances_from_plurality(df, possible_answers_num, exclude=False, weights=None):
    n = df.shape[0]
    distances = np.zeros(n)
    answers = plurality(df, possible_answers_num, weights)

    for i in range(n):
        if exclude:
            answers = plurality(df, possible_answers_num, weights, [i])
        distances[i] = distance.hamming(df.iloc[i][:], answers, w=None)
    return distances
