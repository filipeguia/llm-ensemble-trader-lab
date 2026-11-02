from __future__ import annotations

import pytest

from llm_ensemble_lab.config import CostConfig
from llm_ensemble_lab.cost import (
    PRICING,
    BudgetExceeded,
    BudgetTracker,
    Pricing,
    Usage,
    estimate_run_cost,
    estimate_tokens,
    resolve_pricing,
)


def test_estimate_tokens_is_monotonic_and_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a", 3.0) == 1
    assert estimate_tokens("abcd", 3.0) == 2
    assert estimate_tokens("a" * 300, 3.0) == 100
    assert estimate_tokens("a" * 301, 3.0) == 101
    with pytest.raises(ValueError):
        estimate_tokens("x", 0)


def test_pricing_math_per_million_tokens():
    p = Pricing(input_per_mtok=2.0, output_per_mtok=10.0)
    assert p.cost(Usage(input_tokens=1_000_000)) == pytest.approx(2.0)
    assert p.cost(Usage(output_tokens=1_000_000)) == pytest.approx(10.0)
    assert p.cost(Usage(cache_write_tokens=1_000_000)) == pytest.approx(2.5)
    assert p.cost(Usage(cache_read_tokens=1_000_000)) == pytest.approx(0.2)
    mixed = Usage(input_tokens=500_000, output_tokens=100_000, cache_read_tokens=1_000_000)
    assert p.cost(mixed) == pytest.approx(1.0 + 1.0 + 0.2)


def test_usage_accumulates():
    u = Usage(1, 2, 3, 4)
    u += Usage(10, 20, 30, 40)
    assert u.to_dict() == {
        "input_tokens": 11,
        "output_tokens": 22,
        "cache_write_tokens": 33,
        "cache_read_tokens": 44,
    }


def test_resolve_pricing_table_override_and_unknown():
    cfg = CostConfig()
    assert resolve_pricing("claude-sonnet-5-5", cfg) == PRICING["claude-sonnet-5-5"]
    override = CostConfig(price_input_per_mtok=1.0, price_output_per_mtok=2.0)
    assert resolve_pricing("anything", override) == Pricing(1.0, 2.0)
    with pytest.raises(KeyError, match="no price known"):
        resolve_pricing("some-unlisted-model", cfg)


def _estimate(n_triggers=10, n_agents=5, **cfg_kwargs):
    return estimate_run_cost(
        system_prompt="s" * 300,
        user_prompts=["u" * 600] * n_triggers,
        n_agents=n_agents,
        model="claude-sonnet-5-5",
        cfg=CostConfig(**cfg_kwargs),
        max_tokens=1000,
    )


def test_estimate_counts_calls_and_tokens():
    est = _estimate(n_triggers=10, n_agents=5)
    assert est.n_calls == 50
    # (300 + 600) chars / 3 per token = 300 tokens per call
    assert est.input_tokens == 300 * 50
    assert est.output_tokens == 200 * 50


def test_estimate_scales_linearly_with_agents():
    one = _estimate(n_agents=1)
    eight = _estimate(n_agents=8)
    assert eight.usd_expected == pytest.approx(one.usd_expected * 8)


def test_upper_bound_is_not_below_expected():
    est = _estimate()
    assert est.usd_upper_bound >= est.usd_expected > 0


def test_cache_hit_rate_lowers_the_expected_cost():
    cold = _estimate(cache_hit_rate=0.0)
    warm = _estimate(cache_hit_rate=0.9)
    assert warm.usd_expected < cold.usd_expected
    assert warm.usd_upper_bound == pytest.approx(cold.usd_upper_bound)


def test_estimate_with_no_triggers_is_zero():
    est = estimate_run_cost(
        system_prompt="s", user_prompts=[], n_agents=5, model="claude-sonnet-5-5", cfg=CostConfig(), max_tokens=100
    )
    assert est.n_calls == 0 and est.usd_expected == 0.0


def test_estimate_with_explicit_pricing_does_not_need_a_known_model():
    est = estimate_run_cost(
        system_prompt="s",
        user_prompts=["u"],
        n_agents=1,
        model="unlisted",
        cfg=CostConfig(),
        max_tokens=100,
        pricing=Pricing(1.0, 1.0),
    )
    assert est.model == "unlisted"


def test_budget_tracker_raises_when_the_cap_is_passed():
    tracker = BudgetTracker(Pricing(1.0, 1.0), max_usd=1.0)
    tracker.add(Usage(input_tokens=400_000, output_tokens=400_000))
    assert tracker.spent_usd == pytest.approx(0.8)
    with pytest.raises(BudgetExceeded):
        tracker.add(Usage(input_tokens=200_000, output_tokens=200_000))
    assert tracker.calls == 2


def test_budget_tracker_without_cap_never_raises():
    tracker = BudgetTracker(Pricing(1.0, 1.0), max_usd=None)
    tracker.add(Usage(input_tokens=10_000_000))
    assert tracker.spent_usd == pytest.approx(10.0)
