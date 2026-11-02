"""Ensemble policy: ask N agents, validate every answer, vote

The policy holds no market state. It receives a context dict and the position
flag, queries the backend once per agent (optionally in parallel), turns each
raw answer into a validated Decision or an abstention, and applies the
consensus rule
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .backends.base import Backend, DecisionRequest
from .cost import BudgetTracker, Usage
from .prompts import render_user_prompt
from .schema import (
    Decision,
    DecisionError,
    Limits,
    check_against_context,
    extract_json_object,
    parse_decision,
)
from .voting import ConsensusResult, ConsensusRule, decide_consensus


@dataclass
class MemberResult:
    agent_index: int
    decision: Decision | None
    error: str | None = None
    attempts: int = 1
    usage: Usage = field(default_factory=Usage)

    @property
    def valid(self) -> bool:
        return self.decision is not None

    def to_dict(self) -> dict[str, Any]:
        d = self.decision
        return {
            "agent": self.agent_index,
            "valid": self.valid,
            "action": d.action if d else None,
            "sl_price": d.sl_price if d else None,
            "tp_price": d.tp_price if d else None,
            "error": self.error,
            "attempts": self.attempts,
        }


@dataclass
class EnsembleOutcome:
    consensus: ConsensusResult
    members: list[MemberResult]


class EnsemblePolicy:
    def __init__(
        self,
        backend: Backend,
        system_prompt: str,
        *,
        n_agents: int,
        rule: ConsensusRule,
        limits: Limits | None = None,
        parse_retries: int = 0,
        max_workers: int = 1,
        budget: BudgetTracker | None = None,
    ):
        if n_agents < 1:
            raise ValueError("n_agents must be at least 1")
        self.backend = backend
        self.system_prompt = system_prompt
        self.n_agents = n_agents
        self.rule = rule
        self.limits = limits or Limits()
        self.parse_retries = parse_retries
        self.max_workers = max_workers
        self.budget = budget

    def decide(self, ctx: dict[str, Any]) -> EnsembleOutcome:
        user_prompt = render_user_prompt(ctx)
        ref_price = float(ctx["bar_close"])
        has_position = ctx["position"] is not None
        indices = list(range(self.n_agents))
        if self.max_workers > 1 and self.n_agents > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, self.n_agents)) as pool:
                futures = [
                    pool.submit(self._ask_member, i, ctx, user_prompt, ref_price, has_position)
                    for i in indices
                ]
                members = [f.result() for f in futures]
        else:
            members = [self._ask_member(i, ctx, user_prompt, ref_price, has_position) for i in indices]
        votes = [m.decision for m in members]
        return EnsembleOutcome(consensus=decide_consensus(votes, self.rule), members=members)

    def _ask_member(
        self, agent_index: int, ctx: dict[str, Any], user_prompt: str, ref_price: float, has_position: bool
    ) -> MemberResult:
        usage = Usage()
        last_error: str | None = None
        attempts = 0
        for attempt in range(self.parse_retries + 1):
            attempts += 1
            response = self.backend.decide(
                DecisionRequest(
                    trigger_id=ctx["trigger_id"],
                    agent_index=agent_index,
                    attempt=attempt,
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    context=ctx,
                )
            )
            usage += response.usage
            if self.budget is not None:
                self.budget.add(response.usage)
            if response.error:
                # Transport-level failure, asking again would repeat the same request
                last_error = response.error
                break
            try:
                decision = parse_decision(extract_json_object(response.raw_text))
                check_against_context(
                    decision, ref_price=ref_price, has_position=has_position, limits=self.limits
                )
                return MemberResult(agent_index, decision, None, attempts, usage)
            except DecisionError as exc:
                last_error = str(exc)
        return MemberResult(agent_index, None, last_error, attempts, usage)
