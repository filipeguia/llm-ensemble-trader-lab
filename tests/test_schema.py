from __future__ import annotations

import json

import pytest

from llm_ensemble_lab.schema import (
    Decision,
    DecisionError,
    Limits,
    check_against_context,
    extract_json_object,
    parse_decision,
)


def good(**over):
    base = {"action": "HOLD", "sl_price": None, "tp_price": None, "reasoning": "x"}
    base.update(over)
    return base


# ---- parse_decision: valid answers -------------------------------------------------


@pytest.mark.parametrize("action", ["HOLD", "CLOSE"])
def test_non_entry_actions_parse(action):
    d = parse_decision(good(action=action))
    assert d == Decision(action, None, None, "x")


@pytest.mark.parametrize("action", ["BUY", "SELL"])
def test_entry_actions_parse(action):
    d = parse_decision(good(action=action, sl_price=99, tp_price=101.5))
    assert d.action == action
    assert d.sl_price == 99.0 and d.tp_price == 101.5


def test_reasoning_is_optional_and_truncated():
    assert parse_decision({"action": "HOLD"}).reasoning == ""
    long = parse_decision(good(reasoning="a" * 500))
    assert len(long.reasoning) == 200


# ---- parse_decision: malformed answers ---------------------------------------------


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({}, "missing field: action"),
        (good(action="buy"), "action must be one of"),
        (good(action="LONG"), "action must be one of"),
        (good(action=None), "action must be one of"),
        (good(action=1), "action must be one of"),
        (good(action="BUY"), "requires sl_price and tp_price"),
        (good(action="BUY", sl_price=99), "requires sl_price and tp_price"),
        (good(action="SELL", tp_price=99), "requires sl_price and tp_price"),
        (good(action="HOLD", sl_price=99), "to be null"),
        (good(action="CLOSE", tp_price=101), "to be null"),
        (good(action="BUY", sl_price="99", tp_price=101), "must be a number or null"),
        (good(action="BUY", sl_price=True, tp_price=101), "must be a number or null"),
        (good(action="BUY", sl_price=-1, tp_price=101), "must be positive"),
        (good(action="BUY", sl_price=0, tp_price=101), "must be positive"),
        (good(action="BUY", sl_price=float("nan"), tp_price=101), "finite"),
        (good(action="BUY", sl_price=float("inf"), tp_price=101), "finite"),
        (good(reasoning=123), "reasoning must be a string"),
        (good(extra="field"), "unknown field"),
    ],
)
def test_malformed_decisions_are_rejected(payload, fragment):
    with pytest.raises(DecisionError) as err:
        parse_decision(payload)
    assert fragment in str(err.value)


def test_non_mapping_is_rejected():
    with pytest.raises(DecisionError):
        parse_decision(["HOLD"])  # type: ignore[arg-type]


# ---- extract_json_object -----------------------------------------------------------


def test_extract_bare_object():
    assert extract_json_object('{"action": "HOLD"}') == {"action": "HOLD"}


def test_extract_from_code_fence():
    text = 'Here you go\n```json\n{"action": "HOLD", "sl_price": null}\n```\nthanks'
    assert extract_json_object(text)["action"] == "HOLD"


def test_extract_from_prose_prefix():
    assert extract_json_object('My answer: {"action": "CLOSE"}')["action"] == "CLOSE"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I would hold",
        "[1, 2, 3]",
        '"just a string"',
        '{"action": "HOLD",}',
        '{"action": "HOLD"',
        '{"action": NaN}',
        '{"action": Infinity}',
    ],
)
def test_extract_rejects_non_objects_and_bad_json(text):
    with pytest.raises(DecisionError):
        extract_json_object(text)


def test_extract_non_string_input():
    with pytest.raises(DecisionError):
        extract_json_object(None)  # type: ignore[arg-type]


def test_roundtrip_through_json():
    d = Decision("BUY", 99.0, 101.0, "ok")
    assert parse_decision(extract_json_object(json.dumps(d.to_dict()))) == d


# ---- check_against_context ---------------------------------------------------------


def test_buy_geometry():
    d = Decision("BUY", 99.0, 102.0)
    check_against_context(d, ref_price=100.0, has_position=False)
    with pytest.raises(DecisionError):
        check_against_context(Decision("BUY", 101.0, 102.0), ref_price=100.0, has_position=False)
    with pytest.raises(DecisionError):
        check_against_context(Decision("BUY", 99.0, 99.5), ref_price=100.0, has_position=False)


def test_sell_geometry():
    check_against_context(Decision("SELL", 101.0, 98.0), ref_price=100.0, has_position=False)
    with pytest.raises(DecisionError):
        check_against_context(Decision("SELL", 99.0, 98.0), ref_price=100.0, has_position=False)
    with pytest.raises(DecisionError):
        check_against_context(Decision("SELL", 101.0, 100.5), ref_price=100.0, has_position=False)


def test_position_state_rules():
    with pytest.raises(DecisionError, match="already open"):
        check_against_context(Decision("BUY", 99.0, 102.0), ref_price=100.0, has_position=True)
    with pytest.raises(DecisionError, match="no position"):
        check_against_context(Decision("CLOSE"), ref_price=100.0, has_position=False)
    check_against_context(Decision("CLOSE"), ref_price=100.0, has_position=True)
    check_against_context(Decision("HOLD"), ref_price=100.0, has_position=True)
    check_against_context(Decision("HOLD"), ref_price=100.0, has_position=False)


def test_limits_are_applied():
    wide = Decision("BUY", 80.0, 120.0)
    with pytest.raises(DecisionError, match="stop distance"):
        check_against_context(wide, ref_price=100.0, has_position=False, limits=Limits(max_stop_pct=10))
    check_against_context(wide, ref_price=100.0, has_position=False, limits=Limits(max_stop_pct=25))
    poor = Decision("BUY", 99.0, 100.5)
    with pytest.raises(DecisionError, match="reward/risk"):
        check_against_context(poor, ref_price=100.0, has_position=False, limits=Limits(min_reward_risk=1.0))
    check_against_context(poor, ref_price=100.0, has_position=False, limits=Limits(min_reward_risk=0.5))
