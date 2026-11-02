"""Compare a full ensemble (for example N=8) with smaller sub-ensembles (N=4)

The question: how much does the consensus change when fewer agents vote?

Method, with no new model calls:
1. A run queries all N agents at every trigger and records each answer
   (decisions.jsonl)
2. Every size-k subset of the N agents is re-voted on the recorded answers,
   using the same rule scaled to k (an absolute quorum_votes is scaled to the
   same share of k, rounded up)
3. Each subset's consensus is compared with the full ensemble's consensus,
   trigger by trigger

Reported per subset size: number of entry signals, agreement with the full
ensemble, and how many triggers flip for at least one subset. The comparison
holds the market context fixed at each trigger (the context seen in the
reference run), so it measures voting stability only, it does not re-simulate
trades
"""
from __future__ import annotations

import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .schema import ENTRY_ACTIONS, Decision
from .voting import ConsensusRule, decide_consensus


@dataclass(frozen=True)
class VoteRecord:
    trigger_id: int
    votes: tuple[Decision | None, ...]


def load_vote_records(path: str | Path) -> list[VoteRecord]:
    records: list[VoteRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            members = sorted(row["members"], key=lambda m: m["agent"])
            votes = tuple(
                Decision(m["action"], m["sl_price"], m["tp_price"]) if m["valid"] else None
                for m in members
            )
            records.append(VoteRecord(row["trigger_id"], votes))
    return records


def _spread(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _choose_subsets(n: int, k: int, max_combos: int, seed: int) -> tuple[list[tuple[int, ...]], bool]:
    total = math.comb(n, k)
    if total <= max_combos:
        return list(itertools.combinations(range(n), k)), False
    rng = random.Random(seed)
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < max_combos:
        chosen.add(tuple(sorted(rng.sample(range(n), k))))
    return sorted(chosen), True


def compare_sub_ensembles(
    records: Sequence[VoteRecord],
    rule: ConsensusRule,
    sub_size: int,
    *,
    max_combos: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("no recorded triggers to compare")
    n = len(records[0].votes)
    if any(len(r.votes) != n for r in records):
        raise ValueError("all records must have the same number of agents")
    if not 1 <= sub_size < n:
        raise ValueError(f"sub_size must be between 1 and {n - 1}")

    reference = [decide_consensus(r.votes, rule).action for r in records]
    sub_rule = rule
    if rule.quorum_votes is not None:
        # An absolute quorum keeps its share of the ensemble, rounded up, so it scales to the subset size
        scaled = max(1, math.ceil(min(rule.quorum_votes, n) * sub_size / n - 1e-9))
        sub_rule = replace(rule, quorum_votes=scaled)
    subsets, sampled = _choose_subsets(n, sub_size, max_combos, seed)

    entries: list[float] = []
    agreement: list[float] = []
    flips = [0] * len(records)
    for subset in subsets:
        actions = [
            decide_consensus([r.votes[a] for a in subset], sub_rule).action for r in records
        ]
        entries.append(sum(1 for a in actions if a in ENTRY_ACTIONS))
        same = [a == ref for a, ref in zip(actions, reference)]
        agreement.append(sum(same) / len(same))
        for t, ok in enumerate(same):
            if not ok:
                flips[t] += 1

    flip_rates = [f / len(subsets) for f in flips]
    return {
        "n_agents": n,
        "sub_size": sub_size,
        "n_triggers": len(records),
        "required_full": rule.required_votes(n),
        "required_sub": sub_rule.required_votes(sub_size),
        "n_subsets_total": math.comb(n, sub_size),
        "n_subsets_used": len(subsets),
        "subsets_sampled": sampled,
        "reference_entry_signals": sum(1 for a in reference if a in ENTRY_ACTIONS),
        "sub_entry_signals": _spread(entries),
        "sub_agreement_with_reference": _spread(agreement),
        "triggers_with_any_flip": sum(1 for f in flips if f > 0),
        "mean_flip_rate": statistics.fmean(flip_rates),
    }


def format_comparison(result: dict[str, Any], label: str = "") -> str:
    r = result
    head = (
        f"{label}full N={r['n_agents']} needs {r['required_full']} votes, "
        f"sub-ensembles N={r['sub_size']} need {r['required_sub']} votes, "
        f"{r['n_subsets_used']} of {r['n_subsets_total']} subsets"
        + (" (sampled)" if r["subsets_sampled"] else "")
    )
    a = r["sub_agreement_with_reference"]
    e = r["sub_entry_signals"]
    lines = [
        head,
        f"  triggers evaluated            : {r['n_triggers']}",
        f"  entry signals, full ensemble  : {r['reference_entry_signals']}",
        f"  entry signals, sub-ensembles  : mean {e['mean']:.2f}  std {e['std']:.2f}  min {e['min']:.0f}  max {e['max']:.0f}",
        f"  agreement with full ensemble  : mean {a['mean']:.3f}  std {a['std']:.3f}  min {a['min']:.3f}  max {a['max']:.3f}",
        f"  triggers that flip in any subset: {r['triggers_with_any_flip']}  (mean flip rate {r['mean_flip_rate']:.3f})",
    ]
    return "\n".join(lines)
