import numpy as np
import pandas as pd

from scipy.stats import chi2
from configurations.constants import Constants

"""
Based on the pseudo-code in: Qi Li1 et. al., "A Confidence-Aware Approach for Truth Discovery on Long-Tail Data"
Link: https://cse.buffalo.edu/~jing/doc/vldb15_CATD.pdf

INPUT: Continuous data
PARAMS: 
        - iterations: # of repetitions of estimating the answers and weighting the workers by the estimations.
        - alpha: by default 0.05, a most used confidence interval coverage 
"""


def CATD(iterations, data: pd.DataFrame, alpha=0.05):
    n, m = data.shape
    weights = np.ones(n)
    df_values = data.values
    agg_answers = np.average(df_values, axis=0, weights=weights, returned=False).reshape(1, -1)

    for iteration in range(iterations):
        ssr = ((df_values - agg_answers)**2).sum(axis=1)
        ssr = np.clip(ssr, a_min=Constants.fault_min, a_max=Constants.fault_max)
        w_i_t = chi2.ppf(alpha/2, m) / ssr
        w_i_t = w_i_t / w_i_t.sum()
        agg_answers = np.average(df_values, axis=0, weights=w_i_t.reshape(-1, ), returned=False)

    return agg_answers
