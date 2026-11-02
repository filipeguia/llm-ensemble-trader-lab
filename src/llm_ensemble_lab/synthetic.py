"""Synthetic OHLC generator: geometric Brownian motion with regime switches

The output has the shape of hourly bars but no relation to any real market.
Regimes differ in drift and volatility and switch with a Markov chain, which
produces enough volatility jumps to exercise the trigger and the agents
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .bars import Bar


@dataclass(frozen=True)
class Regime:
    drift: float  # mean log return per bar
    vol: float  # standard deviation of the log return per bar


REGIMES: dict[str, Regime] = {
    "calm": Regime(drift=0.0, vol=0.0008),
    "trend_up": Regime(drift=0.0004, vol=0.0012),
    "trend_down": Regime(drift=-0.0004, vol=0.0012),
    "volatile": Regime(drift=0.0, vol=0.0030),
}


def generate_with_regimes(
    n_bars: int,
    seed: int = 7,
    start: str | datetime = "2024-01-01T00:00:00",
    bar_minutes: int = 60,
    start_price: float = 100.0,
    substeps: int = 8,
    persistence: float = 0.97,
) -> tuple[list[Bar], list[str]]:
    """Return (bars, regime name per bar), fully determined by the seed"""
    if n_bars < 1:
        raise ValueError("n_bars must be at least 1")
    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    rng = random.Random(seed)
    t0 = start if isinstance(start, datetime) else datetime.fromisoformat(start)
    names = list(REGIMES)
    regime = "calm"
    price = round(start_price, 6)
    bars: list[Bar] = []
    labels: list[str] = []
    for k in range(n_bars):
        if rng.random() > persistence:
            regime = rng.choice([r for r in names if r != regime])
        reg = REGIMES[regime]
        sub_vol = reg.vol / math.sqrt(substeps)
        sub_drift = (reg.drift - 0.5 * reg.vol**2) / substeps
        open_ = price
        p = open_
        hi = lo = p
        for _ in range(substeps):
            p *= math.exp(sub_drift + sub_vol * rng.gauss(0.0, 1.0))
            hi = max(hi, p)
            lo = min(lo, p)
        close = round(p, 6)
        bars.append(
            Bar(
                time=t0 + timedelta(minutes=bar_minutes * k),
                open=open_,
                high=round(max(hi, open_, close), 6),
                low=round(min(lo, open_, close), 6),
                close=close,
            )
        )
        labels.append(regime)
        price = close
    return bars, labels


def generate_bars(n_bars: int, seed: int = 7, **kwargs) -> list[Bar]:
    return generate_with_regimes(n_bars, seed=seed, **kwargs)[0]
