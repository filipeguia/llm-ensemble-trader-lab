"""Command line interface

    llm-lab generate   write a synthetic OHLC CSV
    llm-lab estimate   count triggers and estimate the cost, nothing is called
    llm-lab run        run a backtest (use --dry-run first)
    llm-lab compare    compare N agents with smaller sub-ensembles on a finished run
    llm-lab aggregate  aggregate summaries of several runs
    llm-lab show-config  print the default configuration as JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .backends.base import BackendFatalError
from .bars import DataError, write_bars_csv
from .config import ConfigError, RunConfig, config_from_dict
from .ensemble_compare import compare_sub_ensembles, format_comparison, load_vote_records
from .results import aggregate_summaries, format_aggregate, load_summary
from .runner import CostCapError, estimate_cost, format_estimate, prepare, run_experiment
from .synthetic import generate_bars
from .voting import ConsensusRule


def _set_path(data: dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise ConfigError(f"--set {dotted}: {k!r} is not a config section")
        node = node[k]
    if keys[-1] not in node:
        raise ConfigError(f"--set {dotted}: unknown key {keys[-1]!r}")
    try:
        node[keys[-1]] = json.loads(raw)
    except json.JSONDecodeError:
        node[keys[-1]] = raw


def _build_config(args: argparse.Namespace) -> RunConfig:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        base = RunConfig().to_dict()
        for section, values in data.items():
            if isinstance(values, dict) and isinstance(base.get(section), dict):
                base[section].update(values)
            else:
                base[section] = values
        data = base
    else:
        data = RunConfig().to_dict()
    if getattr(args, "csv", None):
        data["data"]["csv"] = args.csv
    if getattr(args, "backend", None):
        data["backend"]["kind"] = args.backend
    if getattr(args, "agents", None):
        data["ensemble"]["n_agents"] = args.agents
    if getattr(args, "model", None):
        data["backend"]["model"] = args.model
    if getattr(args, "exchange_dir", None):
        data["backend"]["exchange_dir"] = args.exchange_dir
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_path(data, key, value)
    return config_from_dict(data)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON file with config overrides, see configs/default.json")
    p.add_argument("--csv", help="OHLC CSV (time,open,high,low,close), default is synthetic data")
    p.add_argument("--backend", choices=["mock", "api", "file"], help="execution backend")
    p.add_argument("--agents", type=int, help="ensemble size N")
    p.add_argument("--model", help="model id for the api backend")
    p.add_argument("--exchange-dir", help="exchange directory for the file backend")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config value, e.g. trigger.lookback=24")


def cmd_generate(args: argparse.Namespace) -> int:
    bars = generate_bars(args.bars, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_bars_csv(out, bars)
    print(f"wrote {len(bars)} synthetic bars to {out}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    prepared = prepare(cfg)
    print(f"bars: {len(prepared.bars)}, trigger events: {len(prepared.events)}")
    print(format_estimate(estimate_cost(cfg, prepared), cfg, cfg.backend.kind))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    summary = run_experiment(cfg, args.out, dry_run=args.dry_run, force=args.force)
    print(
        f"done: {summary['n_triggers_consulted']} triggers consulted, "
        f"{summary['n_consensus_entries']} consensus entries, {summary['n_trades']} trades, "
        f"{summary['n_invalid_answers']} invalid agent answers, spent ${summary['spent_usd']:.4f}"
    )
    print(f"results in {args.out}")
    return 0 if summary["completed"] else 3


def cmd_compare(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = config_from_dict(json.load(fh))
    records = load_vote_records(run_dir / "decisions.jsonl")
    if not records:
        raise ConfigError("decisions.jsonl has no records")
    base_rule = ConsensusRule.from_config(cfg.ensemble.rule)
    results = []
    rules = [(None, base_rule)]
    if args.fractions:
        rules = [
            (
                f,
                ConsensusRule.from_config(
                    replace(cfg.ensemble.rule, mode="quorum", quorum_fraction=f, quorum_votes=None)
                ),
            )
            for f in args.fractions
        ]
    for fraction, rule in rules:
        res = compare_sub_ensembles(
            records, rule, args.sub_size, max_combos=args.max_combos, seed=args.seed
        )
        label = f"[quorum_fraction={fraction}] " if fraction is not None else f"[{rule.mode}] "
        print(format_comparison(res, label))
        results.append({"quorum_fraction": fraction, "mode": rule.mode, **res})
    out = run_dir / "ensemble_compare.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(f"saved {out}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    summaries = [load_summary(d) for d in args.run_dirs]
    agg = aggregate_summaries(summaries)
    print(format_aggregate(agg))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2)
            fh.write("\n")
        print(f"saved {args.out}")
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    print(json.dumps(RunConfig().to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write a synthetic OHLC CSV")
    p.add_argument("--out", default="data/sample_synthetic_bars.csv")
    p.add_argument("--bars", type=int, default=720)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("estimate", help="count triggers and estimate cost, no calls")
    _add_common(p)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("run", help="run a backtest")
    _add_common(p)
    p.add_argument("--out", default="runs/run01", help="output directory for result files")
    p.add_argument("--dry-run", action="store_true", help="full pipeline with the mock backend, no key and no network")
    p.add_argument("--force", action="store_true", help="overwrite result files in a non-empty output directory")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare the full ensemble with smaller sub-ensembles")
    p.add_argument("run_dir")
    p.add_argument("--sub-size", type=int, default=4)
    p.add_argument("--max-combos", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fractions", type=float, nargs="+", help="sweep quorum fractions instead of using the run's rule")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("aggregate", help="aggregate summary.json files of several runs")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", help="write the aggregate as JSON")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("show-config", help="print the default configuration")
    p.set_defaults(func=cmd_show_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DataError, BackendFatalError, CostCapError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
