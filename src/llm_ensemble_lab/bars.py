"""OHLC bar container, CSV input and output, validation and return statistics"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, cast

CSV_COLUMNS = ("time", "open", "high", "low", "close")


class DataError(ValueError):
    """Raised when a price series is malformed"""


@dataclass(frozen=True, slots=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float

    def iso(self) -> str:
        return self.time.isoformat(timespec="minutes")


def validate_bars(bars: Sequence[Bar]) -> None:
    """Check ordering, finiteness and the OHLC envelope of every bar"""
    previous: datetime | None = None
    for i, b in enumerate(bars):
        values = (b.open, b.high, b.low, b.close)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise DataError(f"bar {i}: prices must be finite and positive")
        if b.high < max(b.open, b.close) or b.low > min(b.open, b.close) or b.high < b.low:
            raise DataError(f"bar {i}: high/low do not contain open and close")
        if previous is not None and b.time <= previous:
            raise DataError(f"bar {i}: timestamps must be strictly increasing")
        previous = b.time


def load_bars_csv(path: str | Path, start: str | None = None, end: str | None = None) -> list[Bar]:
    """Read bars from a CSV with the columns time,open,high,low,close"""
    bars: list[Bar] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"CSV is missing column(s): {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                bars.append(
                    Bar(
                        time=datetime.fromisoformat(row["time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise DataError(f"CSV line {line_no}: {exc}") from exc
    if start is not None:
        lo = datetime.fromisoformat(start)
        bars = [b for b in bars if b.time >= lo]
    if end is not None:
        hi = datetime.fromisoformat(end)
        bars = [b for b in bars if b.time <= hi]
    if not bars:
        raise DataError("no bars left after loading and filtering")
    validate_bars(bars)
    return bars


def write_bars_csv(path: str | Path, bars: Sequence[Bar]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for b in bars:
            writer.writerow([b.time.isoformat(timespec="minutes"), b.open, b.high, b.low, b.close])


def log_returns(bars: Sequence[Bar]) -> list[float | None]:
    """Log return of each close against the previous close

    Entry 0 is None because the first bar has no previous close. The value at i
    only depends on bars i-1 and i
    """
    out: list[float | None] = [None] * len(bars)
    for i in range(1, len(bars)):
        out[i] = math.log(bars[i].close / bars[i - 1].close)
    return out


def trailing_return_stats(bars: Sequence[Bar], lookback: int) -> list[tuple[float, float] | None]:
    """Mean and sample standard deviation of the `lookback` log returns before each bar

    The window for bar t holds the returns of bars t-lookback .. t-1, so bar t
    never enters its own statistics. Entry t is None during warm-up, until
    `lookback` returns exist before t. The value at t only depends on bars up to t-1
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    rets = log_returns(bars)
    out: list[tuple[float, float] | None] = [None] * len(bars)
    for t in range(lookback + 1, len(bars)):
        window = cast(list[float], rets[t - lookback : t])  # all defined, t - lookback >= 1
        mean = statistics.fmean(window)
        out[t] = (mean, statistics.stdev(window, mean))
    return out
