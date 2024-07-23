import numpy as np

import measures.vector_distance_functions

# To make usable, uncomment all lines with #**#
from nltk.translate.gleu_score import sentence_gleu
from itertools import combinations_with_replacement



def gleu_distance(x: np.array, y: np.array):
    if len(x) == 1:
        # TODO: ugly hack to handle single-column datasets
        x = x[0]
    if  len(y) == 1:
        y = y[0]
    return 1 - (sentence_gleu([x.split(" ")], y.split(" ")) + sentence_gleu([y.split(" ")], x.split(" "))) / 2


def gleu_all_pairs(df):
    df_frame = df #.to_frame()
    answers = df_frame.shape[0]
    indices = list(combinations_with_replacement(range(answers), 2))

    result = np.zeros(shape=(answers, answers))
    distance = None
    for i, j in indices:
        value1 = df_frame.loc[i][0]
        value2 = df_frame.loc[j][0]
        distance = measures.vector_distance_functions.gleu_distance(value1, value2)
        result[i, j] = result[j, i] = distance

    return result
