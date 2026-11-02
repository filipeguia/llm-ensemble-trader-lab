"""Token usage, pricing and the pre-run cost estimator

The price table is a snapshot of list prices used for illustration only. Prices
change, so every value can be overridden from the configuration, and an unknown
model without explicit prices is an error instead of a silent zero
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import CostConfig


class BudgetExceeded(RuntimeError):
    """Raised when accumulated spend reaches the configured cap"""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        return self

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens, cache multipliers apply to the input price"""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_mult: float = 1.25
    cache_read_mult: float = 0.10

    def cost(self, usage: Usage) -> float:
        per = 1_000_000.0
        return (
            usage.input_tokens * self.input_per_mtok / per
            + usage.output_tokens * self.output_per_mtok / per
            + usage.cache_write_tokens * self.input_per_mtok * self.cache_write_mult / per
            + usage.cache_read_tokens * self.input_per_mtok * self.cache_read_mult / per
        )


# Illustrative snapshot, verify against the provider pricing page before relying on it
PRICING_SNAPSHOT_NOTE = "illustrative list prices, check the provider pricing page"
PRICING: dict[str, Pricing] = {
    "claude-sonnet-5-5": Pricing(2.00, 10.00),
    "claude-opus-5-5": Pricing(4.00, 20.00),
    "claude-haiku-5-5": Pricing(0.10, 0.50),
    "mock": Pricing(0.0, 0.0),
}


def resolve_pricing(model: str, cfg: CostConfig) -> Pricing:
    """Explicit prices in the config win over the table"""
    if cfg.price_input_per_mtok is not None and cfg.price_output_per_mtok is not None:
        return Pricing(cfg.price_input_per_mtok, cfg.price_output_per_mtok)
    if model in PRICING:
        return PRICING[model]
    raise KeyError(
        f"no price known for model {model!r}: set cost.price_input_per_mtok and "
        "cost.price_output_per_mtok"
    )


def estimate_tokens(text: str, chars_per_token: float = 3.0) -> int:
    """Rough token count from character length

    A deliberately conservative heuristic (numeric text tokenizes densely). Use
    the provider token counting endpoint when an exact figure matters
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return math.ceil(len(text) / chars_per_token)


@dataclass(frozen=True)
class CostEstimate:
    n_triggers: int
    n_agents: int
    n_calls: int
    input_tokens: int
    output_tokens: int
    usd_expected: float
    usd_upper_bound: float
    model: str

    def to_dict(self) -> dict[str, object]:
        return {
            "n_triggers": self.n_triggers,
            "n_agents": self.n_agents,
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd_expected": round(self.usd_expected, 4),
            "usd_upper_bound": round(self.usd_upper_bound, 4),
            "model": self.model,
            "pricing_note": PRICING_SNAPSHOT_NOTE,
        }


def estimate_run_cost(
    *,
    system_prompt: str,
    user_prompts: Sequence[str],
    n_agents: int,
    model: str,
    cfg: CostConfig,
    max_tokens: int,
    pricing: Pricing | None = None,
) -> CostEstimate:
    """Estimate the spend of a run before any call is made

    `user_prompts` holds one rendered prompt per trigger. Every agent receives
    the same prompt at a trigger, so calls = triggers x agents. Expected output
    uses cfg.assumed_output_tokens, the upper bound assumes every call uses the
    full max_tokens. When `pricing` is not given it is resolved from the model
    name and the cost config
    """
    pricing = pricing or resolve_pricing(model, cfg)
    sys_tokens = estimate_tokens(system_prompt, cfg.chars_per_token)
    n_triggers = len(user_prompts)
    in_tokens = 0
    for prompt in user_prompts:
        in_tokens += (sys_tokens + estimate_tokens(prompt, cfg.chars_per_token)) * n_agents
    n_calls = n_triggers * n_agents
    out_expected = cfg.assumed_output_tokens * n_calls
    out_upper = max_tokens * n_calls

    cached = int(in_tokens * cfg.cache_hit_rate)
    fresh = in_tokens - cached
    expected = pricing.cost(Usage(input_tokens=fresh, output_tokens=out_expected, cache_read_tokens=cached))
    upper = pricing.cost(Usage(input_tokens=in_tokens, output_tokens=out_upper))
    return CostEstimate(
        n_triggers=n_triggers,
        n_agents=n_agents,
        n_calls=n_calls,
        input_tokens=in_tokens,
        output_tokens=out_expected,
        usd_expected=expected,
        usd_upper_bound=upper,
        model=model,
    )


class BudgetTracker:
    """Thread-safe running total of spend with a hard cap"""

    def __init__(self, pricing: Pricing, max_usd: float | None):
        self.pricing = pricing
        self.max_usd = max_usd
        self.usage = Usage()
        self.spent_usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, usage: Usage) -> None:
        with self._lock:
            self.usage += usage
            self.calls += 1
            self.spent_usd = self.pricing.cost(self.usage)
            if self.max_usd is not None and self.spent_usd > self.max_usd:
                raise BudgetExceeded(
                    f"spend ${self.spent_usd:.4f} passed the cap of ${self.max_usd:.2f} after {self.calls} calls"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "usage": self.usage.to_dict(),
            "spent_usd": round(self.spent_usd, 6),
            "max_usd": self.max_usd,
        }


def total_usage(usages: Iterable[Usage]) -> Usage:
    total = Usage()
    for u in usages:
        total += u
    return total
