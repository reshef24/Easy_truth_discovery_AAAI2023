from .voting_rules import (
    vr_random_dictator,
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
    bitmap_majority,
)


class VotingRules:
    _rules = {
        rule.__name__: rule
        for rule in (
            vr_random_dictator,
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
            bitmap_majority,
        )
    }

    @staticmethod
    def get_voting_rule(name: str):
        if name not in VotingRules._rules:
            valid_names = ", ".join(VotingRules._rules.keys())
            raise ValueError(f"Invalid voting rule name. Valid names: {valid_names}")
        return VotingRules._rules[name]
