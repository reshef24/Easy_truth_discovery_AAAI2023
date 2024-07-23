import numpy as np
import pandas as pd

from configurations.constants import Constants

"""
Based on the pseudo-code in: Bo Zhao Jiawei Han, "A Probabilistic Model for Estimating Real-valued Truth from
Conflicting Sources"
Link: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.301.2071&rep=rep1&type=pdf
INPUT: Continuous data # TODO check if it's true 
PARAMS:
    - iterations: # of repetitions of E and M steps for the use EM algorithm.
    - alpha, beta: parameters of the variance - prior distribution.
    - mu0, sigma0: parameters of the truth - prior distribution.
"""


def GTM(data: pd.DataFrame, iterations, alpha=10, beta=20, sigma0=1, mu0=0):
    n, m = data.shape
    df_values = data.values

    # Normalize the data
    weights = np.ones(n)
    truth_prior = np.median(df_values, axis=0).reshape(1, -1)
    sigma_prior = np.clip(np.std(df_values, axis=0).reshape(1, -1),
                          a_min=Constants.fault_min, a_max=Constants.fault_max)
    normalized_df_values = (df_values - truth_prior) / sigma_prior

    agg_answers = np.average(normalized_df_values, axis=0, weights=weights, returned=False).reshape(1, -1)
    for iteration in range(iterations):
        ssr = ((normalized_df_values - agg_answers)**2).sum(axis=1).reshape(-1, 1)
        sigma_i = (2 * beta + ssr) / (2 * (alpha + 1) + m)
        agg_answers = (mu0 / sigma0 + (normalized_df_values / sigma_i).sum(axis=0)) / (1 / sigma0 + (1 / sigma_i).sum())

    # Denormalize the answers
    denormalized_answers = (agg_answers * sigma_prior + truth_prior).reshape(-1)
    return denormalized_answers
