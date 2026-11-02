from __future__ import annotations

import json

import pytest

from llm_ensemble_lab.backends.base import BackendResponse, DecisionRequest
from llm_ensemble_lab.config import ConfigError, RunConfig, config_from_dict, load_config, save_config
from llm_ensemble_lab.cost import BudgetExceeded, BudgetTracker, Pricing, Usage
from llm_ensemble_lab.ensemble import EnsemblePolicy
from llm_ensemble_lab.prompts import build_system_prompt, render_user_prompt
from llm_ensemble_lab.config import PromptConfig
from llm_ensemble_lab.voting import ConsensusRule

CTX = {
    "trigger_id": 1,
    "trigger": {"name": "return_zscore", "direction": "up", "strength": 3.1},
    "bar_index": 30,
    "bar_time": "2024-01-01T15:00",
    "bar_close": 100.0,
    "ret_std": 0.005,
    "price_vol": 0.5,
    "recent_bars": [{"time": "2024-01-01T14:00", "open": 99.5, "high": 100.2, "low": 99.4, "close": 100.0}],
    "position": None,
    "unrealized_pnl": None,
    "policy_hints": {"stop_vol_mult": 2.0, "target_vol_mult": 2.0},
}

BUY_JSON = json.dumps({"action": "BUY", "sl_price": 99.5, "tp_price": 100.5, "reasoning": "r"})
HOLD_JSON = json.dumps({"action": "HOLD", "sl_price": None, "tp_price": None, "reasoning": "r"})


class ScriptedBackend:
    """Returns the scripted answers for each agent in order, per attempt"""

    name = "scripted"

    def __init__(self, answers):
        self.answers = answers  # {agent_index: [raw_text or BackendResponse, ...]}
        self.calls: list[tuple[int, int]] = []

    def decide(self, request: DecisionRequest) -> BackendResponse:
        self.calls.append((request.agent_index, request.attempt))
        item = self.answers[request.agent_index][request.attempt]
        if isinstance(item, BackendResponse):
            return item
        return BackendResponse(raw_text=item, usage=Usage(input_tokens=100, output_tokens=10))


def policy_for(backend, n=3, **kw):
    return EnsemblePolicy(
        backend, "SYS", n_agents=n, rule=ConsensusRule(mode="majority"), **kw
    )


# ---- ensemble ----------------------------------------------------------------------


def test_ensemble_votes_over_valid_answers():
    backend = ScriptedBackend({0: [BUY_JSON], 1: [BUY_JSON], 2: [HOLD_JSON]})
    out = policy_for(backend).decide(CTX)
    assert out.consensus.action == "BUY"
    assert [m.valid for m in out.members] == [True, True, True]


def test_invalid_answers_abstain_and_are_recorded():
    backend = ScriptedBackend({0: [BUY_JSON], 1: ["not json at all"], 2: ['{"action": "LONG"}']})
    out = policy_for(backend).decide(CTX)
    assert out.consensus.action == "HOLD"
    assert [m.valid for m in out.members] == [True, False, False]
    assert out.members[1].error and out.members[2].error


def test_context_violations_are_invalid_votes():
    # BUY with the stop above the price is structurally valid but impossible in context
    bad_geometry = json.dumps({"action": "BUY", "sl_price": 101.0, "tp_price": 102.0})
    backend = ScriptedBackend({0: [bad_geometry], 1: [bad_geometry], 2: [BUY_JSON]})
    out = policy_for(backend).decide(CTX)
    assert [m.valid for m in out.members] == [False, False, True]
    assert out.consensus.action == "HOLD"


def test_parse_retry_asks_again_after_an_invalid_answer():
    backend = ScriptedBackend({0: ["oops", BUY_JSON], 1: [BUY_JSON], 2: [BUY_JSON]})
    out = policy_for(backend, parse_retries=1).decide(CTX)
    assert out.members[0].valid and out.members[0].attempts == 2
    assert (0, 1) in backend.calls and (1, 1) not in backend.calls


def test_transport_errors_are_not_retried():
    failing = BackendResponse(error="RateLimitError: slow down")
    backend = ScriptedBackend({0: [failing, BUY_JSON], 1: [BUY_JSON], 2: [BUY_JSON]})
    out = policy_for(backend, parse_retries=2).decide(CTX)
    assert not out.members[0].valid and out.members[0].attempts == 1
    assert out.members[0].error == "RateLimitError: slow down"


def test_budget_is_charged_and_enforced():
    backend = ScriptedBackend({0: [BUY_JSON], 1: [BUY_JSON], 2: [BUY_JSON]})
    budget = BudgetTracker(Pricing(1000.0, 1000.0), max_usd=0.2)
    with pytest.raises(BudgetExceeded):
        policy_for(backend, budget=budget).decide(CTX)


def test_parallel_workers_give_the_same_result_as_sequential():
    answers = {i: [BUY_JSON if i % 2 == 0 else HOLD_JSON] for i in range(5)}
    seq = policy_for(ScriptedBackend(answers), n=5).decide(CTX)
    par = policy_for(ScriptedBackend(answers), n=5, max_workers=4).decide(CTX)
    assert seq.consensus.to_dict() == par.consensus.to_dict()
    assert [m.agent_index for m in par.members] == list(range(5))


def test_policy_rejects_empty_ensemble():
    with pytest.raises(ValueError):
        policy_for(ScriptedBackend({}), n=0)


# ---- prompts -----------------------------------------------------------------------


def test_system_prompt_contains_contract_and_toy_policy_label():
    text = build_system_prompt(PromptConfig(stop_vol_mult=2.0, target_vol_mult=3.0))
    assert '"action"' in text and "sl_price" in text and "tp_price" in text
    assert "not a trading strategy" in text
    assert "2.0 times price_vol" in text and "3.0 times price_vol" in text
    assert "toy rule" in text


def test_custom_policy_file_replaces_the_toy_block(tmp_path):
    policy = tmp_path / "policy.txt"
    policy.write_text('Always answer HOLD. Example JSON: {"action": "HOLD"}', encoding="utf-8")
    text = build_system_prompt(PromptConfig(policy_file=str(policy)))
    assert "Always answer HOLD" in text and "placeholder" not in text


def test_user_prompt_renders_flat_and_open_states():
    flat = render_user_prompt(CTX)
    assert "Position: NONE" in flat and "time,open,high,low,close" in flat
    held = dict(CTX, position={"side": "long", "entry_price": 99.0, "sl_price": 98.0, "tp_price": 101.0, "bars_held": 3}, unrealized_pnl=1.0)
    assert "Position: LONG" in render_user_prompt(held)


def test_agent_templates_match_the_decision_contract():
    from pathlib import Path

    agents = Path(__file__).resolve().parents[1] / "agents"
    for name in ("toy-decision-agent-exchange.md", "toy-decision-agent-single-shot.md"):
        text = (agents / name).read_text(encoding="utf-8")
        for key in ("action", "sl_price", "tp_price", "reasoning", "BUY", "SELL", "HOLD", "CLOSE"):
            assert key in text, f"{name} is missing {key}"


# ---- config ------------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    cfg = RunConfig()
    path = tmp_path / "c.json"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_partial_config_dict_fills_defaults():
    cfg = config_from_dict({"ensemble": {"n_agents": 9}})
    assert cfg.ensemble.n_agents == 9
    assert cfg.trigger == RunConfig().trigger


def test_defaults_are_hourly_bars_zscore_trigger_and_simple_majority():
    cfg = RunConfig()
    assert cfg.data.synthetic.bar_minutes == 60
    assert (cfg.trigger.name, cfg.trigger.lookback, cfg.trigger.z_threshold) == ("return_zscore", 48, 2.5)
    assert cfg.ensemble.rule.mode == "quorum" and cfg.ensemble.rule.quorum_fraction == 0.5


def test_shipped_default_config_matches_the_code_defaults():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    assert load_config(path) == RunConfig()


@pytest.mark.parametrize(
    "bad",
    [
        {"nope": 1},
        {"ensemble": {"n_agents": 0}},
        {"ensemble": {"rule": {"mode": "plurality"}}},
        {"ensemble": {"rule": {"quorum_fraction": 1.5}}},
        {"ensemble": {"rule": {"priority": ["BUY", "SELL"]}}},
        {"backend": {"kind": "carrier-pigeon"}},
        {"cost": {"cache_hit_rate": 2}},
        {"trigger": {"lookback": 1}},
        {"trigger": {"z_threshold": 0}},
        {"prompt": {"stop_vol_mult": 0}},
        {"data": "not an object"},
    ],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)
