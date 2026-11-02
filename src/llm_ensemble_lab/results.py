"""Result files: trade CSV, per-run summary and aggregation across runs"""
from __future__ import annotations

import csv
import dataclasses
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backtest import Trade

TRADE_FIELDS = [f.name for f in dataclasses.fields(Trade)]


def write_trades_csv(path: str | Path, trades: Sequence[Trade]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        for t in trades:
            writer.writerow(dataclasses.asdict(t))


def summarize_trades(trades: Sequence[Trade], starting_equity: float) -> dict[str, Any]:
    """Per-run trade statistics, in the synthetic quote currency of the run"""
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    curve = starting_equity
    peak = starting_equity
    max_dd = 0.0
    for p in pnls:
        curve += p
        peak = max(peak, curve)
        max_dd = max(max_dd, peak - curve)
    return {
        "n_trades": n,
        "n_long": sum(1 for t in trades if t.side == "long"),
        "n_short": sum(1 for t in trades if t.side == "short"),
        "win_rate": (sum(1 for p in pnls if p > 0) / n) if n else 0.0,
        "total_pnl": sum(pnls),
        "avg_pnl": (sum(pnls) / n) if n else 0.0,
        "best_trade": max(pnls) if n else 0.0,
        "worst_trade": min(pnls) if n else 0.0,
        "max_drawdown": max_dd,
        "final_equity": starting_equity + sum(pnls),
        "exit_reasons": dict(Counter(t.exit_reason for t in trades)),
    }


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean, std, median, min and max of every numeric field shared by all runs"""
    if not summaries:
        raise ValueError("no summaries to aggregate")
    shared = set(summaries[0])
    for s in summaries[1:]:
        shared &= set(s)
    out: dict[str, Any] = {"n_runs": len(summaries), "metrics": {}}
    for key in sorted(shared):
        values = [s[key] for s in summaries]
        if not all(_is_number(v) for v in values):
            continue
        out["metrics"][key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return out


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    with open(Path(run_dir) / "summary.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_aggregate(agg: Mapping[str, Any]) -> str:
    lines = [f"runs aggregated: {agg['n_runs']}", f"{'metric':<22}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}"]
    for key, m in agg["metrics"].items():
        lines.append(f"{key:<22}{m['mean']:>12.4f}{m['std']:>12.4f}{m['min']:>12.4f}{m['max']:>12.4f}")
    return "\n".join(lines)
