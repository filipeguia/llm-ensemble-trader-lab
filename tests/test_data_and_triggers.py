from __future__ import annotations

import math
import statistics
from dataclasses import replace

import pytest

from conftest import make_bars
from llm_ensemble_lab.bars import (
    DataError,
    load_bars_csv,
    log_returns,
    trailing_return_stats,
    validate_bars,
    write_bars_csv,
)
from llm_ensemble_lab.config import TriggerConfig
from llm_ensemble_lab.synthetic import REGIMES, generate_bars, generate_with_regimes
from llm_ensemble_lab.triggers import ReturnZScoreTrigger, make_trigger

# ---- bars and return statistics ----------------------------------------------------


def test_log_returns_start_with_none_and_use_the_previous_close():
    bars = make_bars([(10, 11, 9, 10), (10, 12, 10, 11), (11, 11, 9, 9)])
    rets = log_returns(bars)
    assert rets[0] is None
    assert rets[1] == pytest.approx(math.log(11 / 10))
    assert rets[2] == pytest.approx(math.log(9 / 11))


def test_trailing_stats_are_none_during_warm_up_then_match_a_manual_window():
    bars = generate_bars(60, seed=2)
    stats = trailing_return_stats(bars, 5)
    assert stats[:6] == [None] * 6  # five returns must exist before the bar
    rets = log_returns(bars)
    window = rets[1:6]  # returns of bars 1..5 are the history of bar 6
    assert stats[6] == pytest.approx((statistics.fmean(window), statistics.stdev(window)))


def test_trailing_stats_exclude_the_candidate_bar_and_ignore_later_bars():
    bars = generate_bars(80, seed=3)
    base = trailing_return_stats(bars, 10)
    t = 40
    # bar t (and everything after it) can change without touching the stats at t
    changed = list(bars)
    changed[t] = replace(bars[t], close=bars[t].high)
    for k in range(t + 1, len(changed)):
        changed[k] = replace(changed[k], close=changed[k].open)
    assert trailing_return_stats(changed, 10)[t] == base[t]
    # and the stats of a truncated series match the full series up to its end
    assert trailing_return_stats(bars[:t], 10) == base[:t]


def test_trailing_stats_reject_a_tiny_lookback():
    with pytest.raises(ValueError):
        trailing_return_stats(make_bars([(1, 1, 1, 1)] * 5), 1)


def test_validate_bars_catches_broken_data():
    good = make_bars([(10, 11, 9, 10), (10, 11, 9, 10)])
    validate_bars(good)
    with pytest.raises(DataError, match="contain open and close"):
        validate_bars(make_bars([(10, 9, 9, 10)]))
    with pytest.raises(DataError, match="strictly increasing"):
        validate_bars([good[0], good[0]])
    with pytest.raises(DataError, match="finite and positive"):
        validate_bars(make_bars([(float("nan"), 11, 9, 10)]))
    with pytest.raises(DataError, match="finite and positive"):
        validate_bars(make_bars([(0, 11, 0, 10)]))


def test_csv_roundtrip(tmp_path):
    bars = generate_bars(50, seed=1)
    path = tmp_path / "bars.csv"
    write_bars_csv(path, bars)
    loaded = load_bars_csv(path)
    assert loaded == bars


def test_csv_missing_column_and_bad_row(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("time,open,high,low\n2024-01-01T00:00,1,1,1\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing column"):
        load_bars_csv(p)
    p.write_text("time,open,high,low,close\n2024-01-01T00:00,x,1,1,1\n", encoding="utf-8")
    with pytest.raises(DataError, match="line 2"):
        load_bars_csv(p)


def test_csv_date_filter(tmp_path):
    bars = generate_bars(100, seed=2)
    path = tmp_path / "b.csv"
    write_bars_csv(path, bars)
    sub = load_bars_csv(path, start="2024-01-01T10:00", end="2024-01-01T12:00")
    assert sub[0].time.hour == 10 and sub[-1].time.hour == 12
    with pytest.raises(DataError, match="no bars"):
        load_bars_csv(path, start="2030-01-01")


# ---- synthetic generator -----------------------------------------------------------


def test_generator_is_deterministic():
    assert generate_bars(200, seed=5) == generate_bars(200, seed=5)
    assert generate_bars(200, seed=5) != generate_bars(200, seed=6)


def test_generated_bars_are_valid_and_continuous():
    bars = generate_bars(400, seed=3)
    validate_bars(bars)
    assert all(b.open == a.close for a, b in zip(bars, bars[1:]))
    gaps = {(b.time - a.time).total_seconds() for a, b in zip(bars, bars[1:])}
    assert gaps == {3600.0}


def test_generator_switches_regimes():
    _, labels = generate_with_regimes(2000, seed=7)
    assert len(set(labels)) >= 3
    assert set(labels) <= set(REGIMES)


def test_generator_rejects_bad_arguments():
    with pytest.raises(ValueError):
        generate_bars(0)


def test_shipped_sample_csv_loads():
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "data" / "sample_synthetic_bars.csv"
    bars = load_bars_csv(sample)
    assert len(bars) == 720
    assert bars == generate_bars(720, seed=7)


# ---- trigger -----------------------------------------------------------------------

CFG = TriggerConfig(lookback=10, z_threshold=2.5, cooldown_bars=3)


def series(n, spikes=None, noise=0.001, start=100.0):
    """Bars whose returns alternate by +noise and -noise, with chosen bar returns replaced

    `spikes` maps a bar index to the simple return of that bar. Bar 0 closes at `start`
    """
    spikes = spikes or {}
    closes = [start]
    for k in range(1, n):
        r = spikes.get(k, noise if k % 2 == 0 else -noise)
        closes.append(closes[-1] * (1 + r))
    rows = []
    prev = start
    for c in closes:
        rows.append((prev, max(prev, c), min(prev, c), c))
        prev = c
    return make_bars(rows)


def test_spike_fires_with_direction_and_z_strength():
    up = ReturnZScoreTrigger(CFG).detect(series(60, {40: 0.03}))
    assert [(e.bar_index, e.direction) for e in up] == [(40, "up")]
    assert up[0].strength >= CFG.z_threshold

    down = ReturnZScoreTrigger(CFG).detect(series(60, {40: -0.03}))
    assert [(e.bar_index, e.direction) for e in down] == [(40, "down")]


def test_strength_is_the_absolute_z_score_of_the_bar():
    bars = series(60, {40: 0.03})
    (event,) = ReturnZScoreTrigger(CFG).detect(bars)
    rets = log_returns(bars)
    window = rets[30:40]
    z = (rets[40] - statistics.fmean(window)) / statistics.stdev(window)
    assert event.strength == pytest.approx(abs(z))


def test_flat_data_does_not_fire():
    flat = make_bars([(100.0, 100.0, 100.0, 100.0)] * 80)
    assert ReturnZScoreTrigger(CFG).detect(flat) == []


def test_ordinary_noise_does_not_fire():
    assert ReturnZScoreTrigger(CFG).detect(series(200)) == []


def test_a_window_without_spread_is_guarded_not_divided_by():
    # perfectly flat history, then a large jump: the standard deviation is zero, so no event
    flat_then_jump = make_bars([(100.0, 100.0, 100.0, 100.0)] * 30 + [(100.0, 105.0, 100.0, 105.0)] * 1)
    assert ReturnZScoreTrigger(CFG).detect(flat_then_jump) == []


def test_threshold_decides_whether_a_spike_fires():
    bars = series(60, {40: 0.03})
    assert ReturnZScoreTrigger(replace(CFG, z_threshold=1000.0)).detect(bars) == []
    assert len(ReturnZScoreTrigger(replace(CFG, z_threshold=2.5)).detect(bars)) == 1


def test_trigger_needs_a_full_window_of_returns_before_the_bar():
    lookback = CFG.lookback
    # returns exist from bar 1, so lookback returns are available before bar lookback + 1
    early = ReturnZScoreTrigger(CFG).detect(series(40, {lookback: 0.03}))
    assert early == []
    ok = ReturnZScoreTrigger(CFG).detect(series(40, {lookback + 1: 0.03}))
    assert [e.bar_index for e in ok] == [lookback + 1]


def test_cooldown_blocks_close_events():
    bars = series(60, {40: 0.10, 41: 0.10})
    free = ReturnZScoreTrigger(replace(CFG, cooldown_bars=0)).detect(bars)
    assert [e.bar_index for e in free] == [40, 41]
    blocked = ReturnZScoreTrigger(replace(CFG, cooldown_bars=1)).detect(bars)
    assert [e.bar_index for e in blocked] == [40]


def test_a_later_spike_fires_after_the_cooldown_and_the_old_spike_left_the_window():
    bars = series(90, {40: 0.03, 70: -0.03})
    events = ReturnZScoreTrigger(CFG).detect(bars)
    assert [(e.bar_index, e.direction) for e in events] == [(40, "up"), (70, "down")]


def test_detection_is_causal():
    """Appending future bars must not change earlier events"""
    bars = generate_bars(600, seed=9)
    trig = ReturnZScoreTrigger(TriggerConfig())
    full = trig.detect(bars)
    cut = 350
    head = trig.detect(bars[:cut])
    assert [e for e in full if e.bar_index < cut] == head
    assert head  # the check is not vacuous


def test_changing_future_bars_does_not_change_earlier_events():
    bars = generate_bars(600, seed=9)
    trig = ReturnZScoreTrigger(TriggerConfig())
    cut = 350
    scrambled = list(bars[:cut])
    for b in bars[cut:]:
        scrambled.append(replace(b, high=b.high * 1.5, close=b.high * 1.5))
    before = [e for e in trig.detect(bars) if e.bar_index < cut]
    after = [e for e in trig.detect(scrambled) if e.bar_index < cut]
    assert before == after


def test_trigger_never_fires_before_enough_history():
    bars = generate_bars(200, seed=4)
    cfg = TriggerConfig()
    events = ReturnZScoreTrigger(cfg).detect(bars)
    assert all(e.bar_index >= cfg.lookback + 1 for e in events)


def test_make_trigger_registry():
    assert isinstance(make_trigger(TriggerConfig()), ReturnZScoreTrigger)
    with pytest.raises(ValueError, match="unknown trigger"):
        make_trigger(TriggerConfig(name="nope"))


def test_default_config_produces_events_on_synthetic_data():
    events = make_trigger(TriggerConfig()).detect(generate_bars(1500, seed=7))
    assert 10 < len(events) < 300
