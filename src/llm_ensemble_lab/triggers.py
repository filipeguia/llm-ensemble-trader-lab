"""Event triggers: decide on which bars an agent is consulted

The shipped trigger is a toy: a bar fires when its log return is unusually large
compared with the returns just before it. Parameters come from TriggerConfig and
are placeholders. Detection is causal, an event at bar i only uses data up to bar i
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .bars import Bar, log_returns, trailing_return_stats
from .config import TriggerConfig

# Windows whose standard deviation is below this carry no scale to measure against
# (this also absorbs floating point noise on perfectly flat or perfectly steady data)
MIN_STD = 1e-12


@dataclass(frozen=True)
class TriggerEvent:
    bar_index: int
    direction: str  # "up" or "down"
    strength: float  # absolute z-score of the bar's log return


class Trigger(Protocol):
    name: str

    def detect(self, bars: Sequence[Bar]) -> list[TriggerEvent]: ...


class ReturnZScoreTrigger:
    """Fire when the log return of a bar is `z_threshold` or more standard deviations from the recent mean

    For bar t the mean and the standard deviation come from the `lookback` log
    returns strictly before t, so the candidate never influences its own
    threshold. The event is "up" for a positive z-score and "down" for a negative
    one. No event fires during warm-up or when the window has no spread. After
    an event, the next `cooldown_bars` bars cannot fire
    """

    name = "return_zscore"

    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg

    def detect(self, bars: Sequence[Bar]) -> list[TriggerEvent]:
        cfg = self.cfg
        rets = log_returns(bars)
        stats = trailing_return_stats(bars, cfg.lookback)
        events: list[TriggerEvent] = []
        last: int | None = None
        for i in range(cfg.lookback + 1, len(bars)):
            window_stats = stats[i]
            ret = rets[i]
            if window_stats is None or ret is None:
                continue
            mean, std = window_stats
            if std <= MIN_STD:
                continue
            if last is not None and i - last <= cfg.cooldown_bars:
                continue
            z = (ret - mean) / std
            if abs(z) >= cfg.z_threshold:
                events.append(TriggerEvent(i, "up" if z > 0 else "down", abs(z)))
                last = i
        return events


TRIGGERS: dict[str, type] = {ReturnZScoreTrigger.name: ReturnZScoreTrigger}


def make_trigger(cfg: TriggerConfig) -> Trigger:
    try:
        return TRIGGERS[cfg.name](cfg)
    except KeyError:
        raise ValueError(f"unknown trigger {cfg.name!r}, available: {sorted(TRIGGERS)}") from None
