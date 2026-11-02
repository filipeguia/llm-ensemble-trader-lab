"""Run configuration as plain dataclasses, loadable from and savable to JSON

Every tunable value of the harness lives here so a run is fully described by one
JSON document. All defaults are illustrative placeholders for the toy example,
not tuned values
"""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised for unknown keys or invalid values in a configuration"""


@dataclass
class SyntheticConfig:
    n_bars: int = 1500
    seed: int = 7
    start: str = "2024-01-01T00:00:00"
    bar_minutes: int = 60
    start_price: float = 100.0


@dataclass
class DataConfig:
    # Path to an OHLC CSV (time,open,high,low,close). When None, synthetic bars are generated
    csv: str | None = None
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass
class TriggerConfig:
    name: str = "return_zscore"
    # Number of log returns before the candidate bar used for the mean and the standard deviation
    lookback: int = 48
    # A bar fires when abs(z-score of its log return) reaches this value
    z_threshold: float = 2.5
    cooldown_bars: int = 8


@dataclass
class PromptConfig:
    context_bars: int = 24
    price_decimals: int = 4
    # Toy rule substituted into the policy text: stop and target sit this many
    # multiples of (standard deviation of returns x price) away from the close
    stop_vol_mult: float = 2.0
    target_vol_mult: float = 2.0
    # Optional path to a text file that replaces the toy policy block of the system prompt
    policy_file: str | None = None


@dataclass
class SimConfig:
    position_size: float = 1.0
    half_spread_bps: float = 1.0
    starting_equity: float = 100000.0


@dataclass
class LimitsConfig:
    # Context-dependent checks applied to every agent decision
    max_stop_pct: float = 10.0
    min_reward_risk: float = 0.0


@dataclass
class RuleConfig:
    # "majority": strictly more than half of all members
    # "quorum": quorum_votes, or strictly more than quorum_fraction * n members (capped at n)
    # The defaults are the same simple majority
    mode: str = "quorum"
    quorum_fraction: float = 0.5
    quorum_votes: int | None = None
    # "hold": a tie between leading actions yields HOLD, "priority": first match in `priority` wins
    tie_policy: str = "hold"
    priority: list[str] = field(default_factory=lambda: ["HOLD", "CLOSE", "SELL", "BUY"])
    # "abstain": invalid votes are not counted, "hold": invalid votes count as HOLD
    invalid_vote_policy: str = "abstain"
    # Below this share of valid votes the ensemble answers HOLD
    min_valid_fraction: float = 0.5


@dataclass
class EnsembleConfig:
    n_agents: int = 5
    max_workers: int = 1
    # Extra attempts when an agent answer fails validation
    parse_retries: int = 0
    rule: RuleConfig = field(default_factory=RuleConfig)


@dataclass
class BackendConfig:
    # "mock", "api" or "file"
    kind: str = "mock"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    # Passed as output_config.effort, None omits it
    effort: str | None = "low"
    # Ask the API for schema-constrained JSON through output_config.format
    structured_output: bool = False
    # Opt in to server-side fallback routing on refusals (Claude API only)
    server_side_fallback: bool = False
    request_timeout_s: float = 60.0
    sdk_max_retries: int = 2
    exchange_dir: str = "exchange"
    poll_interval_s: float = 1.0
    wait_timeout_s: float = 900.0
    # A decision file is read once it has been unchanged for this long
    settle_s: float = 0.2
    mock_seed: int = 0
    mock_deviation_rate: float = 0.2
    mock_malformed_rate: float = 0.05


@dataclass
class CostConfig:
    # Hard cap: a run is refused when the estimate exceeds it and aborted when spend reaches it
    max_usd: float = 5.0
    chars_per_token: float = 3.0
    assumed_output_tokens: int = 200
    # Share of input tokens assumed to be served from the prompt cache
    cache_hit_rate: float = 0.0
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> "RunConfig":
        _validate(self)
        return self


def _build(cls: type, data: Mapping[str, Any], path: str) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {path or 'config'}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path + '.' if path else ''}{name} must be an object")
            kwargs[name] = _build(hint, value, f"{path + '.' if path else ''}{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any]) -> RunConfig:
    return _build(RunConfig, data, "").validate()


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return config_from_dict(json.load(fh))


def save_config(cfg: RunConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
        fh.write("\n")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _validate(cfg: RunConfig) -> None:
    s = cfg.data.synthetic
    _require(s.n_bars >= 10, "data.synthetic.n_bars must be at least 10")
    _require(s.bar_minutes >= 1, "data.synthetic.bar_minutes must be positive")
    _require(s.start_price > 0, "data.synthetic.start_price must be positive")

    t = cfg.trigger
    _require(t.lookback >= 2, "trigger.lookback must be at least 2")
    _require(t.z_threshold > 0, "trigger.z_threshold must be positive")
    _require(t.cooldown_bars >= 0, "trigger.cooldown_bars must not be negative")

    p = cfg.prompt
    _require(p.context_bars >= 2, "prompt.context_bars must be at least 2")
    _require(p.stop_vol_mult > 0 and p.target_vol_mult > 0, "prompt volatility multiples must be positive")

    sim = cfg.sim
    _require(sim.position_size > 0, "sim.position_size must be positive")
    _require(sim.half_spread_bps >= 0, "sim.half_spread_bps must not be negative")

    lim = cfg.limits
    _require(lim.max_stop_pct > 0, "limits.max_stop_pct must be positive")
    _require(lim.min_reward_risk >= 0, "limits.min_reward_risk must not be negative")

    e = cfg.ensemble
    _require(e.n_agents >= 1, "ensemble.n_agents must be at least 1")
    _require(e.max_workers >= 1, "ensemble.max_workers must be at least 1")
    _require(e.parse_retries >= 0, "ensemble.parse_retries must not be negative")
    r = e.rule
    _require(r.mode in ("majority", "quorum"), "ensemble.rule.mode must be 'majority' or 'quorum'")
    _require(0 < r.quorum_fraction <= 1, "ensemble.rule.quorum_fraction must be in (0, 1]")
    _require(r.quorum_votes is None or r.quorum_votes >= 1, "ensemble.rule.quorum_votes must be positive")
    _require(r.tie_policy in ("hold", "priority"), "ensemble.rule.tie_policy must be 'hold' or 'priority'")
    _require(
        r.invalid_vote_policy in ("abstain", "hold"),
        "ensemble.rule.invalid_vote_policy must be 'abstain' or 'hold'",
    )
    _require(0 <= r.min_valid_fraction <= 1, "ensemble.rule.min_valid_fraction must be in [0, 1]")
    _require(
        set(r.priority) == {"BUY", "SELL", "HOLD", "CLOSE"} and len(r.priority) == 4,
        "ensemble.rule.priority must list BUY, SELL, HOLD and CLOSE exactly once",
    )

    b = cfg.backend
    _require(b.kind in ("mock", "api", "file"), "backend.kind must be 'mock', 'api' or 'file'")
    _require(b.max_tokens >= 64, "backend.max_tokens must be at least 64")
    _require(b.poll_interval_s > 0, "backend.poll_interval_s must be positive")
    _require(b.wait_timeout_s > 0, "backend.wait_timeout_s must be positive")
    _require(b.settle_s >= 0, "backend.settle_s must not be negative")
    _require(0 <= b.mock_deviation_rate <= 1, "backend.mock_deviation_rate must be in [0, 1]")
    _require(0 <= b.mock_malformed_rate <= 1, "backend.mock_malformed_rate must be in [0, 1]")

    c = cfg.cost
    _require(c.max_usd >= 0, "cost.max_usd must not be negative")
    _require(c.chars_per_token > 0, "cost.chars_per_token must be positive")
    _require(c.assumed_output_tokens >= 1, "cost.assumed_output_tokens must be positive")
    _require(0 <= c.cache_hit_rate <= 1, "cost.cache_hit_rate must be in [0, 1]")
