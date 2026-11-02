from __future__ import annotations

import pytest

from conftest import make_bars
from llm_ensemble_lab.backtest import run_backtest
from llm_ensemble_lab.backends.base import BackendFatalError
from llm_ensemble_lab.config import PromptConfig, SimConfig
from llm_ensemble_lab.cost import BudgetExceeded
from llm_ensemble_lab.ensemble import EnsembleOutcome
from llm_ensemble_lab.schema import Decision
from llm_ensemble_lab.triggers import TriggerEvent
from llm_ensemble_lab.voting import ConsensusResult

FLAT = (100.0, 100.5, 99.5, 100.0)


def scripted(*decisions):
    """Policy that answers with the given decisions in order, then HOLD"""
    queue = list(decisions)
    seen = []

    def decide(ctx):
        seen.append(ctx)
        d = queue.pop(0) if queue else Decision("HOLD")
        return EnsembleOutcome(ConsensusResult(d, "scripted"), [])

    decide.seen = seen  # type: ignore[attr-defined]
    return decide


def run(rows, decisions, trigger_bars=(14,), spread_bps=0.0, size=1.0):
    bars = make_bars(rows)
    events = [TriggerEvent(i, "up", 1.0) for i in trigger_bars]
    policy = scripted(*decisions)
    res = run_backtest(
        bars,
        events,
        policy,
        sim=SimConfig(position_size=size, half_spread_bps=spread_bps),
        prompt=PromptConfig(context_bars=5),
        vol_lookback=3,
        trigger_name="t",
    )
    return res, policy


def pad(rows_after, n=20):
    return [FLAT] * n + rows_after


def test_buy_fills_at_next_open_and_hits_target():
    rows = pad([(101.0, 103.0, 100.5, 102.5)], n=15)  # bar 15 opens at 101, target 102 inside the range
    res, _ = run(rows, [Decision("BUY", 99.0, 102.0)])
    (t,) = res.trades
    assert t.side == "long" and t.entry_idx == 15
    assert t.entry_price == 101.0 and t.exit_price == 102.0
    assert t.exit_reason == "tp" and t.pnl == pytest.approx(1.0)


def test_sell_hits_stop():
    rows = pad([(100.0, 101.5, 99.8, 101.0)], n=15)
    res, _ = run(rows, [Decision("SELL", 101.0, 98.0)])
    (t,) = res.trades
    assert t.side == "short" and t.exit_reason == "sl"
    assert t.exit_price == 101.0 and t.pnl == pytest.approx(-1.0)


def test_stop_wins_when_both_levels_are_inside_the_bar():
    rows = pad([(100.0, 103.0, 97.0, 100.0)], n=15)
    res, _ = run(rows, [Decision("BUY", 99.0, 102.0)])
    assert res.trades[0].exit_reason == "sl" and res.trades[0].exit_price == 99.0


def test_stop_gapped_through_fills_at_the_open():
    rows = pad([(97.0, 98.0, 96.0, 97.5)], n=15)
    res, _ = run(rows, [Decision("BUY", 99.0, 102.0)])
    t = res.trades[0]
    assert t.exit_reason == "sl" and t.exit_price == 97.0
    assert t.pnl == pytest.approx(0.0)  # entry and exit at the same open


def test_spread_is_paid_on_both_fills():
    rows = pad([(100.0, 103.0, 99.9, 102.0)], n=15)
    res, _ = run(rows, [Decision("BUY", 99.0, 102.0)], spread_bps=10.0)
    t = res.trades[0]
    assert t.entry_price == pytest.approx(100.0 * 1.001)
    assert t.exit_price == pytest.approx(102.0 * 0.999)


def test_agent_close_executes_at_next_open():
    rows = pad([(100.0, 100.4, 99.6, 100.2), (100.5, 100.9, 100.1, 100.7), (100.8, 101.0, 100.4, 100.6)], n=15)
    # trigger 1 at bar 14 opens a long, trigger 2 at bar 16 asks to close
    res, _ = run(rows, [Decision("BUY", 99.0, 110.0), Decision("CLOSE")], trigger_bars=(14, 16))
    t = res.trades[0]
    assert t.exit_reason == "agent_close" and t.exit_idx == 17 and t.exit_price == 100.8


def test_open_position_is_closed_at_the_last_close():
    rows = pad([(100.0, 100.6, 99.8, 100.4)] * 3, n=15)
    res, _ = run(rows, [Decision("BUY", 90.0, 120.0)])
    t = res.trades[0]
    assert t.exit_reason == "end_of_data" and t.exit_idx == len(rows) - 1 and t.exit_price == 100.4
    assert res.completed


def test_last_bar_trigger_is_ignored():
    rows = pad([], n=15)
    res, policy = run(rows, [Decision("BUY", 99.0, 101.0)], trigger_bars=(14,))
    assert res.trades == [] and policy.seen == []


def test_context_carries_position_state_to_later_triggers():
    rows = pad([(100.0, 100.4, 99.6, 100.2)] * 4, n=15)
    res, policy = run(rows, [Decision("BUY", 90.0, 120.0), Decision("HOLD")], trigger_bars=(14, 17))
    first, second = policy.seen
    assert first["position"] is None and first["unrealized_pnl"] is None
    assert second["position"]["side"] == "long" and second["unrealized_pnl"] is not None
    assert [c["trigger_id"] for c in policy.seen] == [1, 2]
    assert [r.position_side for r in res.records] == [None, "long"]


def test_recent_bars_window_ends_at_the_trigger_bar():
    rows = pad([(100.0, 100.4, 99.6, 100.2)] * 3, n=15)
    _, policy = run(rows, [], trigger_bars=(14,))
    ctx = policy.seen[0]
    assert len(ctx["recent_bars"]) == 5
    assert ctx["bar_index"] == 14
    assert ctx["recent_bars"][-1]["time"] == ctx["bar_time"]


@pytest.mark.parametrize("exc", [BudgetExceeded("cap"), BackendFatalError("down")])
def test_fatal_errors_stop_the_run_and_keep_partial_results(exc):
    rows = pad([(100.0, 100.4, 99.6, 100.2)] * 6, n=15)
    bars = make_bars(rows)
    calls = {"n": 0}

    def decide(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return EnsembleOutcome(ConsensusResult(Decision("BUY", 90.0, 120.0), "s"), [])
        raise exc

    res = run_backtest(
        bars,
        [TriggerEvent(14, "up", 1.0), TriggerEvent(17, "up", 1.0)],
        decide,
        sim=SimConfig(half_spread_bps=0.0),
        prompt=PromptConfig(context_bars=5),
        vol_lookback=3,
        trigger_name="t",
    )
    assert not res.completed and type(exc).__name__ in res.aborted_reason
    assert len(res.records) == 1
    assert res.trades[0].exit_reason == "aborted"
