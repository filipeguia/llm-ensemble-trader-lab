"""Run orchestration: data, triggers, cost estimate, backtest, result files"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .backends import make_backend
from .backends.base import Backend
from .backtest import BacktestResult, run_backtest
from .bars import Bar, load_bars_csv, trailing_return_stats, validate_bars
from .config import ConfigError, RunConfig
from .cost import (
    PRICING,
    BudgetTracker,
    CostEstimate,
    Pricing,
    estimate_run_cost,
    resolve_pricing,
)
from .ensemble import EnsemblePolicy
from .prompts import build_context, build_system_prompt, render_user_prompt
from .results import summarize_trades, write_trades_csv
from .schema import Limits
from .synthetic import generate_bars
from .triggers import TriggerEvent, make_trigger
from .voting import ConsensusRule


class CostCapError(RuntimeError):
    """The pre-run estimate is above the configured spending cap"""


@dataclass
class Prepared:
    bars: list[Bar]
    events: list[TriggerEvent]
    system_prompt: str
    trigger_name: str


def load_bars(cfg: RunConfig) -> list[Bar]:
    if cfg.data.csv:
        return load_bars_csv(cfg.data.csv)
    s = cfg.data.synthetic
    bars = generate_bars(
        s.n_bars, seed=s.seed, start=s.start, bar_minutes=s.bar_minutes, start_price=s.start_price
    )
    validate_bars(bars)
    return bars


def prepare(cfg: RunConfig) -> Prepared:
    bars = load_bars(cfg)
    trigger = make_trigger(cfg.trigger)
    return Prepared(
        bars=bars,
        events=trigger.detect(bars),
        system_prompt=build_system_prompt(cfg.prompt),
        trigger_name=trigger.name,
    )


def run_digest(cfg: RunConfig, prepared: Prepared) -> str:
    """Fingerprint of everything that decides which triggers and prompts a run produces"""
    payload = {
        "data": asdict(cfg.data),
        "trigger": asdict(cfg.trigger),
        "prompt": asdict(cfg.prompt),
        "n_agents": cfg.ensemble.n_agents,
        "system_prompt": prepared.system_prompt,
        "n_bars": len(prepared.bars),
        "first_close": prepared.bars[0].close,
        "last_close": prepared.bars[-1].close,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _pricing_for(cfg: RunConfig, kind: str) -> Pricing:
    try:
        return resolve_pricing(cfg.backend.model, cfg.cost)
    except KeyError as exc:
        if kind == "api":
            raise ConfigError(str(exc)) from exc
        # Mock and file runs do not bill through this tool, tokens are still reported
        return PRICING["mock"]


def estimate_cost(cfg: RunConfig, prepared: Prepared, kind: str | None = None) -> CostEstimate:
    """Price the run before it starts, from prompts built for every trigger without a position"""
    kind = kind or cfg.backend.kind
    stats = trailing_return_stats(prepared.bars, cfg.trigger.lookback)
    prompts: list[str] = []
    for n, event in enumerate(prepared.events, start=1):
        window_stats = stats[event.bar_index]
        if window_stats is None or event.bar_index >= len(prepared.bars) - 1:
            continue
        ctx = build_context(
            trigger_id=n,
            trigger_name=prepared.trigger_name,
            event=event,
            bars=prepared.bars,
            ret_std=window_stats[1],
            position=None,
            unrealized_pnl=None,
            cfg=cfg.prompt,
        )
        prompts.append(render_user_prompt(ctx))
    return estimate_run_cost(
        system_prompt=prepared.system_prompt,
        user_prompts=prompts,
        n_agents=cfg.ensemble.n_agents,
        model=cfg.backend.model,
        cfg=cfg.cost,
        max_tokens=cfg.backend.max_tokens,
        pricing=_pricing_for(cfg, kind),
    )


def format_estimate(est: CostEstimate, cfg: RunConfig, kind: str) -> str:
    lines = [
        "cost estimate (illustrative list prices, check the provider pricing page)",
        f"  triggers      : {est.n_triggers}",
        f"  agents        : {est.n_agents}",
        f"  calls         : {est.n_calls}",
        f"  input tokens  : ~{est.input_tokens:,}",
        f"  output tokens : ~{est.output_tokens:,} (assumed {cfg.cost.assumed_output_tokens} per call)",
        f"  expected cost : ${est.usd_expected:.4f} for model {est.model}",
        f"  upper bound   : ${est.usd_upper_bound:.4f} (every call uses max_tokens={cfg.backend.max_tokens})",
    ]
    if kind == "mock":
        lines.append("  mock backend  : no spend, figures above are what an API run would cost")
    elif kind == "file":
        lines.append("  file backend  : spend depends on the external agent, tokens are for sizing only")
    return "\n".join(lines)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def run_experiment(
    cfg: RunConfig,
    out_dir: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    backend: Backend | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run one backtest and write its result files, returns the summary dict"""
    cfg.validate()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} is not empty, pass force to overwrite its result files")

    kind = "mock" if dry_run else cfg.backend.kind
    if dry_run and cfg.backend.kind != "mock":
        echo(f"dry-run: the {cfg.backend.kind!r} backend is replaced by the deterministic mock, no key and no network")

    prepared = prepare(cfg)
    if not prepared.events:
        raise ConfigError("the trigger found no events in this data, adjust the trigger or the data")
    estimate = estimate_cost(cfg, prepared, kind)
    echo(format_estimate(estimate, cfg, kind))
    if kind == "api" and estimate.usd_expected > cfg.cost.max_usd:
        raise CostCapError(
            f"expected cost ${estimate.usd_expected:.2f} is above the cap of ${cfg.cost.max_usd:.2f}, "
            "raise cost.max_usd or reduce agents, bars or context"
        )

    digest = run_digest(cfg, prepared)
    if backend is None:
        backend = make_backend(cfg.backend, kind=kind, run_digest=digest)
    # The mock backend spends nothing, so its tokens are priced at zero. The estimate
    # above still uses the configured model so a dry-run shows what the real run costs
    budget = BudgetTracker(
        PRICING["mock"] if kind == "mock" else _pricing_for(cfg, kind),
        cfg.cost.max_usd if kind == "api" else None,
    )
    if kind == "file":
        echo(
            f"file exchange: waiting for external agents, exchange dir = {cfg.backend.exchange_dir}, "
            f"{cfg.ensemble.n_agents} agents per trigger"
        )

    policy = EnsemblePolicy(
        backend,
        prepared.system_prompt,
        n_agents=cfg.ensemble.n_agents,
        rule=ConsensusRule.from_config(cfg.ensemble.rule),
        limits=Limits(cfg.limits.max_stop_pct, cfg.limits.min_reward_risk),
        parse_retries=cfg.ensemble.parse_retries,
        max_workers=cfg.ensemble.max_workers,
        budget=budget,
    )

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            echo(f"  consulted {done}/{total} triggers")

    result = run_backtest(
        prepared.bars,
        prepared.events,
        policy.decide,
        sim=cfg.sim,
        prompt=cfg.prompt,
        vol_lookback=cfg.trigger.lookback,
        trigger_name=prepared.trigger_name,
        on_progress=progress,
    )
    summary = _write_outputs(cfg, out, prepared, result, estimate, budget, kind, dry_run, digest)
    if not result.completed:
        echo(f"run stopped early: {result.aborted_reason}")
    return summary


def _write_outputs(
    cfg: RunConfig,
    out: Path,
    prepared: Prepared,
    result: BacktestResult,
    estimate: CostEstimate,
    budget: BudgetTracker,
    kind: str,
    dry_run: bool,
    digest: str,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "config.json", cfg.to_dict())
    with open(out / "decisions.jsonl", "w", encoding="utf-8") as fh:
        for rec in result.records:
            fh.write(json.dumps(rec.to_dict()) + "\n")
    write_trades_csv(out / "trades.csv", result.trades)

    members = [m for r in result.records for m in r.outcome.members]
    consensus = [r.outcome.consensus.action for r in result.records]
    summary: dict[str, Any] = {
        "backend": kind,
        "dry_run": dry_run,
        "run_digest": digest,
        "completed": result.completed,
        "aborted_reason": result.aborted_reason,
        "n_bars": len(prepared.bars),
        "n_triggers_detected": len(prepared.events),
        "n_triggers_consulted": len(result.records),
        "n_agent_answers": len(members),
        "n_invalid_answers": sum(1 for m in members if not m.valid),
        "n_backend_calls": sum(m.attempts for m in members),
        "n_consensus_entries": sum(1 for a in consensus if a in ("BUY", "SELL")),
        "n_consensus_close": consensus.count("CLOSE"),
        "n_consensus_hold": consensus.count("HOLD"),
        "spent_usd": round(budget.spent_usd, 6),
        **summarize_trades(result.trades, cfg.sim.starting_equity),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "cost.json", {"estimate": estimate.to_dict(), "actual": budget.to_dict()})
    return summary
