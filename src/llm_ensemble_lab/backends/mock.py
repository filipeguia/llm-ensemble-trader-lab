"""Deterministic mock backend

Used by dry-run and by the tests so the whole pipeline runs with no key and no
network. Every answer is a pure function of (seed, trigger, agent, attempt), so
runs are reproducible on any platform. Agents follow the toy policy of the
default system prompt with seeded noise, and some answers are malformed on
purpose to exercise validation
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from ..config import BackendConfig
from ..cost import Usage, estimate_tokens
from .base import BackendResponse, DecisionRequest


def _rng(seed: int, *parts: int) -> random.Random:
    key = ":".join(str(p) for p in (seed, *parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


_MALFORMED = (
    "I would hold here, the move looks stretched",
    '{"action": "BUY", "sl_price": "99.5", "tp_price": 101.0, "reasoning": "string number"}',
    '{"action": "LONG", "sl_price": null, "tp_price": null, "reasoning": "unknown action"}',
    '{"action": "HOLD", "sl_price": 1.0, "tp_price": null, "reasoning": "hold with a level"}',
    '```json\n{"action": "SELL", "sl_price": 101.0, "tp_price": 99.0,\n```',
)


class MockBackend:
    name = "mock"

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    def decide(self, request: DecisionRequest) -> BackendResponse:
        rng = _rng(self.cfg.mock_seed, request.trigger_id, request.agent_index, request.attempt)
        usage = Usage(
            input_tokens=estimate_tokens(request.system_prompt + request.user_prompt),
            output_tokens=60,
        )
        if rng.random() < self.cfg.mock_malformed_rate:
            return BackendResponse(raw_text=rng.choice(_MALFORMED), usage=usage)
        decision = self._policy(request.context, rng)
        return BackendResponse(raw_text=json.dumps(decision), usage=usage)

    def _policy(self, ctx: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        """Toy policy with noise, mirrors the default policy block of the system prompt

        Stop and target sit a multiple of price_vol (return std times price) from the
        close, a toy rule and not a recommendation
        """
        direction = ctx["trigger"]["direction"]
        bars = ctx["recent_bars"]
        close = ctx["bar_close"]
        price_vol = ctx["price_vol"]
        hints = ctx.get("policy_hints", {})
        stop_mult = float(hints.get("stop_vol_mult", 2.0)) * rng.uniform(0.9, 1.1)
        target_mult = float(hints.get("target_vol_mult", 2.0)) * rng.uniform(0.9, 1.1)

        prev_up = len(bars) >= 2 and bars[-2]["close"] > bars[-2]["open"]
        prev_down = len(bars) >= 2 and bars[-2]["close"] < bars[-2]["open"]
        position = ctx.get("position")

        if position is not None:
            against = (position["side"] == "long" and direction == "down") or (
                position["side"] == "short" and direction == "up"
            )
            base = "CLOSE" if against else "HOLD"
        elif direction == "up" and prev_up:
            base = "BUY"
        elif direction == "down" and prev_down:
            base = "SELL"
        else:
            base = "HOLD"

        # Seeded deviation: some agents disagree with the base answer
        if rng.random() < self.cfg.mock_deviation_rate:
            if base in ("BUY", "SELL"):
                base = "HOLD" if rng.random() < 0.7 else ("SELL" if base == "BUY" else "BUY")
            elif base == "HOLD" and position is None:
                base = "BUY" if direction == "up" else "SELL"

        if base == "BUY":
            return {
                "action": "BUY",
                "sl_price": round(close - stop_mult * price_vol, 6),
                "tp_price": round(close + target_mult * price_vol, 6),
                "reasoning": "mock: toy policy",
            }
        if base == "SELL":
            return {
                "action": "SELL",
                "sl_price": round(close + stop_mult * price_vol, 6),
                "tp_price": round(close - target_mult * price_vol, 6),
                "reasoning": "mock: toy policy",
            }
        return {"action": base, "sl_price": None, "tp_price": None, "reasoning": f"mock: {base.lower()}"}
