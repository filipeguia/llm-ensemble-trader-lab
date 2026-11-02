from __future__ import annotations

import pytest

from llm_ensemble_lab.config import RuleConfig
from llm_ensemble_lab.schema import Decision
from llm_ensemble_lab.voting import ConsensusRule, decide_consensus

BUY = Decision("BUY", 99.0, 102.0)
SELL = Decision("SELL", 101.0, 98.0)
HOLD = Decision("HOLD")
CLOSE = Decision("CLOSE")


def buy(sl=99.0, tp=102.0):
    return Decision("BUY", sl, tp)


# ---- required votes ----------------------------------------------------------------


@pytest.mark.parametrize("n, expected", [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (8, 5), (9, 5), (10, 6)])
def test_majority_needs_more_than_half(n, expected):
    assert ConsensusRule(mode="majority").required_votes(n) == expected


@pytest.mark.parametrize("n, expected", [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (7, 4), (8, 5), (9, 5), (10, 6)])
def test_default_rule_is_a_strict_simple_majority(n, expected):
    rule = ConsensusRule()
    assert rule.mode == "quorum" and rule.quorum_fraction == 0.5
    assert rule.required_votes(n) == expected == ConsensusRule(mode="majority").required_votes(n)
    # strictly more than half, and one vote less would not be
    assert rule.required_votes(n) > n / 2
    assert rule.required_votes(n) - 1 <= n / 2


@pytest.mark.parametrize(
    "fraction, n, expected",
    [(0.4, 5, 3), (0.5, 8, 5), (0.7, 5, 4), (0.7, 8, 6), (0.7, 10, 8), (0.75, 4, 4), (0.75, 8, 7), (1.0, 8, 8)],
)
def test_quorum_fraction_needs_strictly_more_than_the_fraction_and_is_capped_at_n(fraction, n, expected):
    assert ConsensusRule(mode="quorum", quorum_fraction=fraction).required_votes(n) == expected


def test_absolute_quorum_is_capped_at_ensemble_size():
    rule = ConsensusRule(mode="quorum", quorum_votes=5)
    assert rule.required_votes(8) == 5
    assert rule.required_votes(4) == 4


def test_required_votes_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        ConsensusRule().required_votes(0)
    with pytest.raises(ValueError):
        ConsensusRule(mode="plurality").required_votes(5)


# ---- consensus ---------------------------------------------------------------------


def test_clear_majority_wins():
    res = decide_consensus([BUY, BUY, BUY, HOLD, HOLD], ConsensusRule(mode="majority"))
    assert res.action == "BUY" and res.reason == "consensus"
    assert res.counts == {"BUY": 3, "HOLD": 2}


def test_no_consensus_defaults_to_hold():
    res = decide_consensus([BUY, BUY, SELL, SELL, HOLD], ConsensusRule(mode="majority"))
    assert res.action == "HOLD" and res.reason == "no_consensus"


def test_quorum_below_threshold_holds():
    res = decide_consensus([BUY, BUY, HOLD, HOLD, HOLD], ConsensusRule())
    assert res.action == "HOLD"
    # HOLD itself reaches the quorum of 3, so this is a consensus to hold
    assert res.reason == "consensus"


def test_quorum_met_by_entry():
    res = decide_consensus([BUY, BUY, BUY, HOLD, SELL], ConsensusRule())
    assert res.action == "BUY" and res.required == 3


def test_exactly_half_of_eight_is_not_a_majority():
    res = decide_consensus([BUY] * 4 + [SELL] * 4, ConsensusRule())
    assert res.action == "HOLD" and res.reason == "no_consensus" and res.required == 5
    res = decide_consensus([BUY] * 4 + [HOLD] * 4, ConsensusRule())
    assert res.action == "HOLD" and res.reason == "no_consensus"


def test_five_of_eight_is_a_majority():
    res = decide_consensus([BUY] * 5 + [HOLD] * 3, ConsensusRule())
    assert res.action == "BUY" and res.required == 5


def test_invalid_answers_do_not_lower_the_threshold():
    # four valid agreeing votes out of eight members are not more than half of the members
    res = decide_consensus([BUY] * 4 + [None] * 4, ConsensusRule())
    assert res.n_valid == 4 and res.required == 5
    assert res.action == "HOLD" and res.reason == "no_consensus"


def test_entry_levels_are_the_median_of_agreeing_votes():
    votes = [buy(98.0, 103.0), buy(99.0, 102.0), buy(97.0, 110.0), HOLD, HOLD]
    res = decide_consensus(votes, ConsensusRule(mode="majority"))
    assert res.action == "BUY"
    assert res.decision.sl_price == 98.0 and res.decision.tp_price == 103.0


def test_even_count_median_is_between_the_middle_values():
    votes = [buy(98.0, 103.0), buy(99.0, 105.0), HOLD]
    res = decide_consensus(votes, ConsensusRule(mode="quorum", quorum_votes=2))
    assert res.decision.sl_price == 98.5 and res.decision.tp_price == 104.0


def test_close_consensus_has_null_levels():
    res = decide_consensus([CLOSE, CLOSE, CLOSE, HOLD, HOLD], ConsensusRule(mode="majority"))
    assert res.action == "CLOSE"
    assert res.decision.sl_price is None and res.decision.tp_price is None


# ---- ties --------------------------------------------------------------------------


def test_tie_between_entries_holds_by_default():
    rule = ConsensusRule(mode="quorum", quorum_votes=2)
    res = decide_consensus([BUY, BUY, SELL, SELL], rule)
    assert res.action == "HOLD" and res.reason == "tie" and res.tie is True


def test_tie_priority_policy_picks_by_order():
    votes = [BUY, BUY, SELL, SELL]
    prefer_sell = ConsensusRule(mode="quorum", quorum_votes=2, tie_policy="priority", priority=("HOLD", "CLOSE", "SELL", "BUY"))
    prefer_buy = ConsensusRule(mode="quorum", quorum_votes=2, tie_policy="priority", priority=("HOLD", "CLOSE", "BUY", "SELL"))
    assert decide_consensus(votes, prefer_sell).action == "SELL"
    assert decide_consensus(votes, prefer_buy).action == "BUY"
    assert decide_consensus(votes, prefer_buy).reason == "tie_priority"


def test_tie_between_entry_and_hold_prefers_hold_with_default_priority():
    rule = ConsensusRule(mode="quorum", quorum_votes=2, tie_policy="priority")
    res = decide_consensus([BUY, BUY, HOLD, HOLD], rule)
    assert res.action == "HOLD"


def test_strictly_higher_count_beats_a_lower_action_that_also_reaches_quorum():
    rule = ConsensusRule(mode="quorum", quorum_votes=2)
    res = decide_consensus([BUY, BUY, BUY, SELL, SELL], rule)
    assert res.action == "BUY" and res.tie is False


# ---- invalid votes -----------------------------------------------------------------


def test_invalid_votes_abstain_by_default():
    rule = ConsensusRule(mode="majority", min_valid_fraction=0.0)
    res = decide_consensus([BUY, BUY, None, None, None], rule)
    # needs 3 of 5 members, only 2 valid votes exist
    assert res.action == "HOLD" and res.n_valid == 2


def test_invalid_votes_can_count_as_hold():
    rule = ConsensusRule(mode="majority", invalid_vote_policy="hold", min_valid_fraction=0.0)
    res = decide_consensus([BUY, None, None, None, None], rule)
    assert res.action == "HOLD" and res.counts == {"BUY": 1, "HOLD": 4}


def test_too_few_valid_votes_forces_hold():
    rule = ConsensusRule(mode="quorum", quorum_votes=1, min_valid_fraction=0.5)
    res = decide_consensus([BUY, None, None, None, None], rule)
    assert res.action == "HOLD" and res.reason == "insufficient_valid_votes"


def test_all_invalid_holds():
    res = decide_consensus([None, None, None], ConsensusRule(min_valid_fraction=0.0))
    assert res.action == "HOLD" and res.reason == "insufficient_valid_votes"


def test_empty_ensemble_holds():
    res = decide_consensus([], ConsensusRule())
    assert res.action == "HOLD" and res.reason == "no_members"


def test_single_agent_ensemble_follows_the_agent():
    assert decide_consensus([BUY], ConsensusRule(mode="majority")).action == "BUY"
    assert decide_consensus([None], ConsensusRule(mode="majority")).action == "HOLD"


def test_rule_from_config_roundtrip():
    rule = ConsensusRule.from_config(RuleConfig(mode="majority", tie_policy="priority"))
    assert rule.mode == "majority" and rule.tie_policy == "priority"
    assert rule.priority == ("HOLD", "CLOSE", "SELL", "BUY")
