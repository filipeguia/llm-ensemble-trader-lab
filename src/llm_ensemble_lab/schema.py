"""Decision JSON contract, tolerant extraction and strict validation

Three separate steps keep the failure modes visible:

1. extract_json_object: pull one JSON object out of model text (code fences and
   short prose around the object are tolerated)
2. parse_decision: strict structural validation of that object
3. check_against_context: price geometry and position-state rules that need the
   market context
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

ACTIONS = ("BUY", "SELL", "HOLD", "CLOSE")
ENTRY_ACTIONS = ("BUY", "SELL")
MAX_REASONING_CHARS = 200
_ALLOWED_KEYS = {"action", "sl_price", "tp_price", "reasoning"}

DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "sl_price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "tp_price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "sl_price", "tp_price", "reasoning"],
    "additionalProperties": False,
}


class DecisionError(ValueError):
    """The answer is not a valid decision"""


@dataclass(frozen=True)
class Decision:
    action: str
    sl_price: float | None = None
    tp_price: float | None = None
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class Limits:
    """Context-dependent limits, generic placeholders for the toy example"""

    max_stop_pct: float = 10.0
    min_reward_risk: float = 0.0


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _reject_constant(name: str) -> Any:
    raise DecisionError(f"non-finite number {name} is not allowed")


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in `text`

    Accepts a bare object, an object inside a markdown code fence, or an object
    preceded by prose. Anything that does not decode to a JSON object raises
    """
    if not isinstance(text, str) or not text.strip():
        raise DecisionError("empty answer")
    candidate = text.strip()
    fence = _FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    decoder = json.JSONDecoder(parse_constant=_reject_constant)
    try:
        obj, _ = decoder.raw_decode(candidate)
        if not isinstance(obj, dict):
            raise DecisionError(f"top-level JSON must be an object, got {type(obj).__name__}")
        return obj
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    if start < 0:
        raise DecisionError("no JSON object found in answer")
    try:
        obj, _ = decoder.raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise DecisionError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise DecisionError("top-level JSON must be an object")
    return obj


def _number_or_none(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"{name} must be a number or null, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise DecisionError(f"{name} must be finite")
    if value <= 0:
        raise DecisionError(f"{name} must be positive")
    return value


def parse_decision(obj: Mapping[str, Any]) -> Decision:
    """Strict structural validation of a decoded object"""
    if not isinstance(obj, Mapping):
        raise DecisionError("decision must be a JSON object")
    unknown = set(obj) - _ALLOWED_KEYS
    if unknown:
        raise DecisionError(f"unknown field(s): {sorted(unknown)}")
    if "action" not in obj:
        raise DecisionError("missing field: action")
    action = obj["action"]
    if not isinstance(action, str) or action not in ACTIONS:
        raise DecisionError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    sl = _number_or_none("sl_price", obj.get("sl_price"))
    tp = _number_or_none("tp_price", obj.get("tp_price"))
    reasoning = obj.get("reasoning", "")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        raise DecisionError("reasoning must be a string")
    reasoning = reasoning.strip()[:MAX_REASONING_CHARS]

    if action in ENTRY_ACTIONS:
        if sl is None or tp is None:
            raise DecisionError(f"{action} requires sl_price and tp_price")
    elif sl is not None or tp is not None:
        raise DecisionError(f"{action} requires sl_price and tp_price to be null")
    return Decision(action=action, sl_price=sl, tp_price=tp, reasoning=reasoning)


def check_against_context(
    decision: Decision,
    *,
    ref_price: float,
    has_position: bool,
    limits: Limits | None = None,
) -> None:
    """Raise DecisionError when the decision cannot be executed in this context"""
    limits = limits or Limits()
    action = decision.action
    if action in ENTRY_ACTIONS and has_position:
        raise DecisionError(f"{action} rejected: a position is already open")
    if action == "CLOSE" and not has_position:
        raise DecisionError("CLOSE rejected: no position is open")
    if action not in ENTRY_ACTIONS:
        return
    sl, tp = decision.sl_price, decision.tp_price
    assert sl is not None and tp is not None
    if action == "BUY" and not (sl < ref_price < tp):
        raise DecisionError(f"BUY needs sl < {ref_price:.4f} < tp, got sl={sl} tp={tp}")
    if action == "SELL" and not (tp < ref_price < sl):
        raise DecisionError(f"SELL needs tp < {ref_price:.4f} < sl, got tp={tp} sl={sl}")
    risk = abs(ref_price - sl)
    reward = abs(tp - ref_price)
    if risk / ref_price * 100 > limits.max_stop_pct:
        raise DecisionError(f"stop distance exceeds {limits.max_stop_pct}% of price")
    if limits.min_reward_risk > 0 and reward < limits.min_reward_risk * risk:
        raise DecisionError(f"reward/risk below {limits.min_reward_risk}")
