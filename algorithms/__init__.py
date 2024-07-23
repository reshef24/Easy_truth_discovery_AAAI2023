from algorithms.DSTD import DSTD
from algorithms.EVTD import EVTD
from algorithms.BPTD import BPTD
from algorithms.EIPTD import EIPTD
from algorithms.DTD import IDTD_mean, DTD_median, IDTD_ranking, \
    IDTD
from algorithms.CRH import CRH, PMTD
from algorithms.CATD import CATD
from algorithms.PTD import IPTD
from algorithms.UA import UA
from algorithms.GTM import GTM
from algorithms.Top2 import Top2
from algorithms.KDEm_from_git import KDEm_alg
from algorithms.Oracle import Oracle, af_external_fault

from measures import vector_distance_functions
from measures import pairwise_distance_functions


class TruthDiscoveryAlgorithms:
    _algorithms = {
        func.__name__: func
        for func in (
            IPTD,
            Oracle,
            af_external_fault,
            UA,
            DSTD,
            IDTD,
            PMTD,
            Top2,
            EIPTD,
            BPTD,
            EVTD,
            IDTD_mean,
            CRH,
            GTM,
            KDEm_alg,
            CATD,
            DTD_median,
            IDTD_ranking,
            Oracle,
        )
    }

    @staticmethod
    def get_algorithm(name: str):
        if name not in TruthDiscoveryAlgorithms._algorithms:
            valid_names = ", ".join(TruthDiscoveryAlgorithms._algorithms.keys())
            raise ValueError(f"Invalid algorithm name. Valid names: {valid_names}")
        return TruthDiscoveryAlgorithms._algorithms[name]


vector_distance_functions_list = [vector_distance_functions.square_euclidean,
                                  vector_distance_functions.square_euclidean_no_zero,
                                  vector_distance_functions.hamming_binary,
                                  vector_distance_functions.hamming_general,
                                  vector_distance_functions.kendall_tau_distance,
                                  vector_distance_functions.footrule_distance,
                                  vector_distance_functions.spearman_rank_corr,
                                  vector_distance_functions.l1,
                                  vector_distance_functions.square_probability,
                                  vector_distance_functions.jaccard_distance,
                                  vector_distance_functions.gleu_distance
]


class ProxyDistanceFunctions:
    _functions = {
        i.__name__: i
        for i in vector_distance_functions_list
    }

    @staticmethod
    def get_function(name: str):
        if name not in ProxyDistanceFunctions._functions:
            valid_names = ", ".join(ProxyDistanceFunctions._functions.keys())
            raise ValueError(f"Invalid function name. Valid names: {valid_names}")
        return ProxyDistanceFunctions._functions[name]


class FaultDistanceFunctions:
    _functions = {
        i.__name__: i
        for i in [
            vector_distance_functions.square_euclidean,
            vector_distance_functions.spearman_rank_corr,
        ]
    }

    @staticmethod
    def get_function(name: str):
        if name not in FaultDistanceFunctions._functions:
            valid_names = ", ".join(FaultDistanceFunctions._functions.keys())
            raise ValueError(f"Invalid function name. Valid names: {valid_names}")
        return FaultDistanceFunctions._functions[name]


class PairwiseDistanceFunctions:
    _functions = {
        i.__name__: j
        for i, j in zip(
            vector_distance_functions_list,
            [
                pairwise_distance_functions.square_euclidean,
                pairwise_distance_functions.square_euclidean_no_zero,
                pairwise_distance_functions.hamming_binary_pairwise,
                pairwise_distance_functions.hamming_general_pairwise,
                pairwise_distance_functions.hamming_binary_pairwise,
                pairwise_distance_functions.footrule,
                pairwise_distance_functions.spearman_rank_correlation,
                pairwise_distance_functions.l1,
                pairwise_distance_functions.square_probability,
                pairwise_distance_functions.jaccard_distance_pairwise,
                pairwise_distance_functions.gleu_distance_pairwise,
            ],
        )
    }

    @staticmethod
    def get_function(name: str):
        if name not in PairwiseDistanceFunctions._functions:
            valid_names = ", ".join(PairwiseDistanceFunctions._functions.keys())
            raise ValueError(f"Invalid function name. Valid names: {valid_names}")
        return PairwiseDistanceFunctions._functions[name]
