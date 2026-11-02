"""Prompt and context builder

The system prompt has a fixed contract part and a swappable policy block. The
shipped policy block is a toy placeholder so the harness can run end to end.
Replace it with `PromptConfig.policy_file` to try a different policy
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .bars import Bar
from .config import PromptConfig
from .triggers import TriggerEvent

SYSTEM_TEMPLATE = """\
You are a decision agent inside an automated backtest harness.
You are consulted only when an event trigger fires on a price series.
Reply with exactly one JSON object and nothing else

OUTPUT CONTRACT
{{"action": "BUY" | "SELL" | "HOLD" | "CLOSE", "sl_price": number or null, "tp_price": number or null, "reasoning": "at most 200 characters"}}
- BUY and SELL need both sl_price and tp_price. Long: sl_price < close < tp_price. Short: tp_price < close < sl_price
- HOLD and CLOSE need sl_price and tp_price set to null
- BUY and SELL are allowed only when no position is open, CLOSE only when one is open
- No other keys, no markdown, no text outside the JSON object

EXECUTION MODEL
- Orders fill at the open of the next bar, one position at a time
- Use only the bars in the message, assume nothing about other data

POLICY (toy example, replace with your own)
{policy}
"""

DEFAULT_POLICY = """\
This block is a placeholder that exercises the harness. It is not a trading strategy
- If the trigger direction is up and the bar before the trigger bar also closed up, answer BUY
- If the trigger direction is down and the bar before the trigger bar also closed down, answer SELL
- If a position is open and the trigger direction is against it, answer CLOSE
- Otherwise answer HOLD
- For BUY and SELL put the stop {stop_vol_mult} times price_vol and the target {target_vol_mult} times price_vol away from the close (a toy rule, price_vol is given in the context)"""


def load_policy_text(cfg: PromptConfig) -> str:
    if cfg.policy_file:
        text = Path(cfg.policy_file).read_text(encoding="utf-8").strip()
    else:
        text = DEFAULT_POLICY
    # Plain replace instead of str.format so a custom policy file may contain braces
    return text.replace("{stop_vol_mult}", str(cfg.stop_vol_mult)).replace(
        "{target_vol_mult}", str(cfg.target_vol_mult)
    )


def build_system_prompt(cfg: PromptConfig) -> str:
    return SYSTEM_TEMPLATE.format(policy=load_policy_text(cfg))


def build_context(
    *,
    trigger_id: int,
    trigger_name: str,
    event: TriggerEvent,
    bars: Sequence[Bar],
    ret_std: float,
    position: dict[str, Any] | None,
    unrealized_pnl: float | None,
    cfg: PromptConfig,
) -> dict[str, Any]:
    """JSON-serializable snapshot handed to every agent at one trigger

    `ret_std` is the standard deviation of the log returns before the trigger
    bar (the same one the trigger z-score uses). `price_vol` is that value times
    the close, one typical bar move in price units
    """
    d = cfg.price_decimals
    i = event.bar_index
    window = bars[max(0, i - cfg.context_bars + 1) : i + 1]
    cur = bars[i]
    return {
        "trigger_id": trigger_id,
        "trigger": {
            "name": trigger_name,
            "direction": event.direction,
            "strength": round(event.strength, 3),
        },
        "bar_index": i,
        "bar_time": cur.iso(),
        "bar_close": round(cur.close, d),
        "ret_std": round(ret_std, 6),
        "price_vol": round(ret_std * cur.close, 6),
        "recent_bars": [
            {
                "time": b.iso(),
                "open": round(b.open, d),
                "high": round(b.high, d),
                "low": round(b.low, d),
                "close": round(b.close, d),
            }
            for b in window
        ],
        "position": position,
        "unrealized_pnl": None if unrealized_pnl is None else round(unrealized_pnl, d),
        "policy_hints": {
            "stop_vol_mult": cfg.stop_vol_mult,
            "target_vol_mult": cfg.target_vol_mult,
        },
    }


def render_user_prompt(ctx: dict[str, Any]) -> str:
    """Compact text rendering of a context, one prompt per trigger"""
    trig = ctx["trigger"]
    lines = [
        f"Trigger {ctx['trigger_id']}: {trig['name']} fired {trig['direction']} "
        f"(absolute return z-score {trig['strength']}) on bar {ctx['bar_time']}",
        f"Close {ctx['bar_close']}  return std {ctx['ret_std']}  price_vol {ctx['price_vol']}",
    ]
    pos = ctx["position"]
    if pos is None:
        lines.append("Position: NONE")
    else:
        lines.append(
            f"Position: {pos['side'].upper()} entry={pos['entry_price']} sl={pos['sl_price']} "
            f"tp={pos['tp_price']} bars_held={pos['bars_held']} unrealized={ctx['unrealized_pnl']}"
        )
    lines.append("Recent bars, oldest first, trigger bar last:")
    lines.append("time,open,high,low,close")
    for b in ctx["recent_bars"]:
        lines.append(f"{b['time']},{b['open']},{b['high']},{b['low']},{b['close']}")
    lines.append("Answer with the JSON object only")
    return "\n".join(lines)
