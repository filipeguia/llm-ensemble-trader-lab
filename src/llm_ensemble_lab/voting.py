"""Ensemble voting: majority, quorum, ties and invalid votes

A vote is a Decision, or None for a member whose answer was invalid or missing.
The consensus is always a concrete Decision. When it is an entry, the stop and
target are the median of the agreeing members' levels
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .config import RuleConfig
from .schema import ENTRY_ACTIONS, Decision

HOLD = Decision(action="HOLD")


@dataclass(frozen=True)
class ConsensusRule:
    mode: str = "quorum"
    quorum_fraction: float = 0.5
    quorum_votes: int | None = None
    tie_policy: str = "hold"
    priority: tuple[str, ...] = ("HOLD", "CLOSE", "SELL", "BUY")
    invalid_vote_policy: str = "abstain"
    min_valid_fraction: float = 0.5

    @classmethod
    def from_config(cls, cfg: RuleConfig) -> "ConsensusRule":
        return cls(
            mode=cfg.mode,
            quorum_fraction=cfg.quorum_fraction,
            quorum_votes=cfg.quorum_votes,
            tie_policy=cfg.tie_policy,
            priority=tuple(cfg.priority),
            invalid_vote_policy=cfg.invalid_vote_policy,
            min_valid_fraction=cfg.min_valid_fraction,
        )

    def required_votes(self, n_members: int) -> int:
        """Votes an action needs to win in an ensemble of `n_members`

        majority: strictly more than half of all members, n // 2 + 1
        quorum: an absolute count when quorum_votes is set, otherwise strictly
        more than quorum_fraction * n members, capped at n, so the same rule
        scales to other ensemble sizes. The default fraction of 0.5 gives the
        same simple majority as the majority mode (N=8 needs 5 votes, N=4 needs 3)

        The threshold is counted against the full ensemble size, so an invalid
        answer never lowers it
        """
        if n_members < 1:
            raise ValueError("n_members must be at least 1")
        if self.mode == "majority":
            return n_members // 2 + 1
        if self.mode == "quorum":
            if self.quorum_votes is not None:
                return min(self.quorum_votes, n_members)
            return min(n_members, math.floor(self.quorum_fraction * n_members + 1e-9) + 1)
        raise ValueError(f"unknown mode {self.mode!r}")


@dataclass(frozen=True)
class ConsensusResult:
    decision: Decision
    reason: str
    counts: dict[str, int] = field(default_factory=dict)
    n_members: int = 0
    n_valid: int = 0
    required: int = 0
    tie: bool = False

    @property
    def action(self) -> str:
        return self.decision.action

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.decision.action,
            "sl_price": self.decision.sl_price,
            "tp_price": self.decision.tp_price,
            "reason": self.reason,
            "counts": dict(self.counts),
            "n_members": self.n_members,
            "n_valid": self.n_valid,
            "required": self.required,
            "tie": self.tie,
        }


def _hold(reason: str, counts: Counter, n: int, n_valid: int, required: int, tie: bool = False) -> ConsensusResult:
    return ConsensusResult(HOLD, reason, dict(counts), n, n_valid, required, tie)


def decide_consensus(votes: Sequence[Decision | None], rule: ConsensusRule) -> ConsensusResult:
    """Combine member votes into one decision

    Order of checks: enough valid votes, then actions that reach the required
    count, then the highest count among them, then tie handling
    """
    n = len(votes)
    if n == 0:
        return _hold("no_members", Counter(), 0, 0, 0)
    required = rule.required_votes(n)

    effective: list[Decision] = []
    for v in votes:
        if v is not None:
            effective.append(v)
        elif rule.invalid_vote_policy == "hold":
            effective.append(HOLD)
    n_valid = len(effective)
    counts = Counter(v.action for v in effective)

    if n_valid < math.ceil(rule.min_valid_fraction * n - 1e-9) or n_valid == 0:
        return _hold("insufficient_valid_votes", counts, n, n_valid, required)

    reaching = {a: c for a, c in counts.items() if c >= required}
    if not reaching:
        return _hold("no_consensus", counts, n, n_valid, required)

    top = max(reaching.values())
    leaders = [a for a, c in reaching.items() if c == top]
    tie = len(leaders) > 1
    if tie:
        if rule.tie_policy == "hold":
            return _hold("tie", counts, n, n_valid, required, tie=True)
        winner = next(a for a in rule.priority if a in leaders)
    else:
        winner = leaders[0]

    if winner in ENTRY_ACTIONS:
        agreeing = [v for v in effective if v.action == winner]
        sl = statistics.median(v.sl_price for v in agreeing if v.sl_price is not None)
        tp = statistics.median(v.tp_price for v in agreeing if v.tp_price is not None)
        decision = Decision(winner, round(sl, 8), round(tp, 8), f"consensus of {len(agreeing)}/{n}")
    else:
        decision = Decision(winner, reasoning=f"consensus of {counts[winner]}/{n}")
    reason = "tie_priority" if tie else "consensus"
    return ConsensusResult(decision, reason, dict(counts), n, n_valid, required, tie)
