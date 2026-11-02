"""Backtest loop with a deliberately simple fill model

Fill model:
- a decision taken on bar i executes at the open of bar i+1
- every fill pays a half spread (basis points of price) against the trader
- stop and target are checked inside the bar, if both fall inside the bar range
  the stop is assumed to hit first (conservative)
- a stop that is already crossed at the open fills at the open
- one position at a time, an open position is closed at the last close
- no slippage model, no partial fills, no fees beyond the spread
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .backends.base import BackendFatalError
from .bars import Bar, trailing_return_stats
from .config import PromptConfig, SimConfig
from .cost import BudgetExceeded
from .ensemble import EnsembleOutcome
from .prompts import build_context
from .triggers import TriggerEvent


@dataclass
class OpenPosition:
    side: str  # "long" or "short"
    entry_idx: int
    entry_time: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    trigger_id: int


@dataclass(frozen=True)
class Trade:
    trigger_id: int
    side: str
    entry_time: str
    entry_idx: int
    entry_price: float
    exit_time: str
    exit_idx: int
    exit_price: float
    sl_price: float
    tp_price: float
    size: float
    pnl: float
    exit_reason: str
    bars_held: int


@dataclass
class TriggerRecord:
    trigger_id: int
    bar_index: int
    bar_time: str
    direction: str
    position_side: str | None
    outcome: EnsembleOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "bar_index": self.bar_index,
            "bar_time": self.bar_time,
            "direction": self.direction,
            "position_side": self.position_side,
            "consensus": self.outcome.consensus.to_dict(),
            "members": [m.to_dict() for m in self.outcome.members],
        }


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    records: list[TriggerRecord] = field(default_factory=list)
    completed: bool = True
    aborted_reason: str | None = None
    bars_processed: int = 0


def _fill_price(price: float, buying: bool, half_spread_bps: float) -> float:
    adj = half_spread_bps / 10_000.0
    return price * (1 + adj) if buying else price * (1 - adj)


def _pnl(pos: OpenPosition, exit_price: float) -> float:
    raw = exit_price - pos.entry_price if pos.side == "long" else pos.entry_price - exit_price
    return raw * pos.size


def check_sl_tp(bar: Bar, pos: OpenPosition) -> tuple[float, str] | None:
    """Return (raw exit price, reason) if the stop or target is reached inside the bar"""
    if pos.side == "long":
        if bar.open <= pos.sl_price:
            return bar.open, "sl"
        if bar.low <= pos.sl_price:
            return pos.sl_price, "sl"
        if bar.high >= pos.tp_price:
            return pos.tp_price, "tp"
    else:
        if bar.open >= pos.sl_price:
            return bar.open, "sl"
        if bar.high >= pos.sl_price:
            return pos.sl_price, "sl"
        if bar.low <= pos.tp_price:
            return pos.tp_price, "tp"
    return None


def run_backtest(
    bars: Sequence[Bar],
    events: Sequence[TriggerEvent],
    decide: Callable[[dict[str, Any]], EnsembleOutcome],
    *,
    sim: SimConfig,
    prompt: PromptConfig,
    vol_lookback: int,
    trigger_name: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Replay `bars`, consulting `decide` at every trigger event

    `decide` receives the context dict and returns an EnsembleOutcome. The
    context carries the standard deviation of the `vol_lookback` log returns
    before the trigger bar. If `decide` raises BudgetExceeded or
    BackendFatalError the loop stops, an open position is closed at the last
    processed close and the result is marked incomplete
    """
    n = len(bars)
    stats = trailing_return_stats(bars, vol_lookback)
    by_bar = {e.bar_index: e for e in events}
    result = BacktestResult()
    position: OpenPosition | None = None
    pending_open: tuple[str, float, float, int] | None = None
    pending_close = False
    hs = sim.half_spread_bps
    next_trigger_id = 1

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal position
        assert position is not None
        selling = position.side == "long"
        price = _fill_price(raw_price, buying=not selling, half_spread_bps=hs)
        result.trades.append(
            Trade(
                trigger_id=position.trigger_id,
                side=position.side,
                entry_time=position.entry_time,
                entry_idx=position.entry_idx,
                entry_price=position.entry_price,
                exit_time=bars[i].iso(),
                exit_idx=i,
                exit_price=price,
                sl_price=position.sl_price,
                tp_price=position.tp_price,
                size=position.size,
                pnl=_pnl(position, price),
                exit_reason=reason,
                bars_held=i - position.entry_idx,
            )
        )
        position = None

    last_processed = -1
    try:
        for i in range(n):
            bar = bars[i]
            last_processed = i
            if pending_close and position is not None:
                close_position(i, bar.open, "agent_close")
            pending_close = False

            if pending_open is not None and position is None:
                side, sl, tp, tid = pending_open
                entry = _fill_price(bar.open, buying=(side == "long"), half_spread_bps=hs)
                position = OpenPosition(side, i, bar.iso(), entry, sl, tp, sim.position_size, tid)
            pending_open = None

            if position is not None:
                hit = check_sl_tp(bar, position)
                if hit is not None:
                    close_position(i, hit[0], hit[1])

            event = by_bar.get(i)
            stats_now = stats[i]
            if event is not None and stats_now is not None and i < n - 1:
                pos_view = None
                unreal = None
                if position is not None:
                    pos_view = {
                        "side": position.side,
                        "entry_price": round(position.entry_price, prompt.price_decimals),
                        "sl_price": position.sl_price,
                        "tp_price": position.tp_price,
                        "bars_held": i - position.entry_idx,
                    }
                    unreal = _pnl(position, bar.close)
                tid = next_trigger_id
                next_trigger_id += 1
                ctx = build_context(
                    trigger_id=tid,
                    trigger_name=trigger_name,
                    event=event,
                    bars=bars,
                    ret_std=stats_now[1],
                    position=pos_view,
                    unrealized_pnl=unreal,
                    cfg=prompt,
                )
                outcome = decide(ctx)
                result.records.append(
                    TriggerRecord(
                        trigger_id=tid,
                        bar_index=i,
                        bar_time=bar.iso(),
                        direction=event.direction,
                        position_side=position.side if position else None,
                        outcome=outcome,
                    )
                )
                dec = outcome.consensus.decision
                if dec.action == "BUY":
                    pending_open = ("long", float(dec.sl_price), float(dec.tp_price), tid)
                elif dec.action == "SELL":
                    pending_open = ("short", float(dec.sl_price), float(dec.tp_price), tid)
                elif dec.action == "CLOSE":
                    pending_close = True
                if on_progress is not None:
                    on_progress(len(result.records), len(events))
            result.bars_processed = i + 1
    except (BudgetExceeded, BackendFatalError) as exc:
        result.completed = False
        result.aborted_reason = f"{type(exc).__name__}: {exc}"

    if position is not None:
        end = last_processed if last_processed >= 0 else n - 1
        reason = "end_of_data" if result.completed else "aborted"
        close_position(end, bars[end].close, reason)
    return result
