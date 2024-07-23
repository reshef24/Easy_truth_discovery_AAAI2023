import numpy as np
import pandas as pd

from utils.aggregations import aggregate
# Unweighted aggregation


def UA(data: pd.DataFrame, voting_rule, mode, params,gt=None):
    possible_answers = set(np.unique(data.values))
    n = data.shape[0]
    weights = np.ones(n) / n
    answers = aggregate(data, gt, weights, voting_rule, mode, possible_answers, params)
    return answers