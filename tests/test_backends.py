from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from llm_ensemble_lab.backends import (
    ApiBackend,
    BackendFatalError,
    DecisionRequest,
    ExchangeTimeout,
    FileExchangeBackend,
    MockBackend,
    make_backend,
)
from llm_ensemble_lab.backends import api as api_module
from llm_ensemble_lab.config import BackendConfig
from llm_ensemble_lab.schema import DecisionError, extract_json_object, parse_decision


def make_request(trigger_id=1, agent_index=0, attempt=0, direction="up", position=None):
    ctx = {
        "trigger_id": trigger_id,
        "trigger": {"name": "return_zscore", "direction": direction, "strength": 3.0},
        "bar_close": 100.0,
        "ret_std": 0.001,
        "price_vol": 0.1,
        "recent_bars": [
            {"time": "t0", "open": 99.0, "high": 99.6, "low": 98.9, "close": 99.5},
            {"time": "t1", "open": 99.5, "high": 100.2, "low": 99.4, "close": 100.0},
        ],
        "position": position,
        "policy_hints": {"stop_vol_mult": 2.0, "target_vol_mult": 2.0},
    }
    return DecisionRequest(trigger_id, agent_index, attempt, "SYSTEM", "USER", ctx)


# ---- mock --------------------------------------------------------------------------


def test_mock_is_deterministic_per_request():
    be = MockBackend(BackendConfig(mock_seed=3))
    req = make_request(trigger_id=4, agent_index=2)
    assert be.decide(req).raw_text == be.decide(req).raw_text


def test_mock_varies_with_seed_agent_and_attempt():
    texts = set()
    for seed in range(4):
        be = MockBackend(BackendConfig(mock_seed=seed, mock_deviation_rate=0.5))
        for agent in range(6):
            texts.add(be.decide(make_request(agent_index=agent)).raw_text)
    assert len(texts) > 3


def test_mock_answers_parse_when_malformed_rate_is_zero():
    be = MockBackend(BackendConfig(mock_malformed_rate=0.0, mock_deviation_rate=0.5))
    for tid in range(1, 40):
        for direction in ("up", "down"):
            raw = be.decide(make_request(trigger_id=tid, direction=direction)).raw_text
            parse_decision(extract_json_object(raw))


def test_mock_malformed_answers_fail_validation():
    be = MockBackend(BackendConfig(mock_malformed_rate=1.0))
    for tid in range(1, 30):
        raw = be.decide(make_request(trigger_id=tid)).raw_text
        with pytest.raises(DecisionError):
            parse_decision(extract_json_object(raw))


def test_mock_policy_follows_trigger_direction():
    be = MockBackend(BackendConfig(mock_malformed_rate=0.0, mock_deviation_rate=0.0))
    up = json.loads(be.decide(make_request(direction="up")).raw_text)
    assert up["action"] == "BUY" and up["sl_price"] < 100.0 < up["tp_price"]
    held = {"side": "long", "entry_price": 99.0, "sl_price": 98.0, "tp_price": 101.0, "bars_held": 2}
    against = json.loads(be.decide(make_request(direction="down", position=held)).raw_text)
    assert against["action"] == "CLOSE"


def test_make_backend_dispatch(tmp_path):
    assert isinstance(make_backend(BackendConfig(), kind="mock"), MockBackend)
    fx = make_backend(BackendConfig(), kind="file", exchange_dir=tmp_path / "ex", run_digest="abc")
    assert isinstance(fx, FileExchangeBackend)
    with pytest.raises(ValueError):
        make_backend(BackendConfig(), kind="nope")


# ---- api (stubbed client, no network) ----------------------------------------------


class StubMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def make_response(text='{"action": "HOLD", "sl_price": null, "tp_price": null}', stop_reason="end_turn", blocks=None):
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    usage = SimpleNamespace(
        input_tokens=120, output_tokens=30, cache_creation_input_tokens=50, cache_read_input_tokens=None
    )
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, stop_details=None)


def stub_backend(response=None, error=None, **cfg):
    stub = SimpleNamespace(messages=StubMessages(response, error))
    return ApiBackend(BackendConfig(**cfg), client=stub), stub.messages


def test_api_request_shape():
    be, messages = stub_backend(make_response(), model="claude-sonnet-5-5")
    be.decide(make_request())
    kw = messages.calls[0]
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["system"][0]["text"] == "SYSTEM"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"] == {"effort": "low"}
    # newer models reject these, they must never be sent
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in kw


def test_api_structured_output_and_effort_switches():
    be, messages = stub_backend(make_response(), structured_output=True, effort=None)
    be.decide(make_request())
    cfg = messages.calls[0]["output_config"]
    assert "effort" not in cfg
    assert cfg["format"]["type"] == "json_schema"
    assert cfg["format"]["schema"]["additionalProperties"] is False


def test_api_usage_and_text_extraction_ignores_thinking_blocks():
    blocks = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text='{"action": "HOLD"}'),
    ]
    be, _ = stub_backend(make_response(blocks=blocks))
    resp = be.decide(make_request())
    assert resp.error is None
    assert resp.raw_text == '{"action": "HOLD"}'
    assert resp.usage.input_tokens == 120 and resp.usage.output_tokens == 30
    assert resp.usage.cache_write_tokens == 50 and resp.usage.cache_read_tokens == 0


def test_api_refusal_truncation_and_empty_answers_abstain():
    be, _ = stub_backend(make_response(stop_reason="refusal"))
    assert "refusal" in be.decide(make_request()).error
    be, _ = stub_backend(make_response(stop_reason="max_tokens"))
    assert "truncated" in be.decide(make_request()).error
    be, _ = stub_backend(make_response(text="   "))
    assert be.decide(make_request()).error == "empty answer"


def test_api_missing_key_is_a_clear_fatal_error(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BackendFatalError, match="ANTHROPIC_API_KEY"):
        ApiBackend(BackendConfig())


def test_api_key_is_read_only_from_the_env_var(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-value")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    ApiBackend(BackendConfig(request_timeout_s=12.0, sdk_max_retries=1))
    assert seen == {"api_key": "test-key-value", "timeout": 12.0, "max_retries": 1}


def _status_error(cls, status):
    anthropic = pytest.importorskip("anthropic")
    import httpx2 as httpx  # the SDK is built on httpx2

    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    response = httpx.Response(status, request=request)
    return getattr(anthropic, cls)("boom", response=response, body=None)


def test_api_transient_errors_abstain_and_fatal_errors_raise():
    pytest.importorskip("anthropic")
    be, _ = stub_backend(error=_status_error("RateLimitError", 429))
    out = be.decide(make_request())
    assert out.error and "RateLimitError" in out.error

    be, _ = stub_backend(error=_status_error("InternalServerError", 500))
    assert be.decide(make_request()).error

    be, _ = stub_backend(error=_status_error("AuthenticationError", 401))
    with pytest.raises(BackendFatalError, match="AuthenticationError"):
        be.decide(make_request())

    be, _ = stub_backend(error=_status_error("BadRequestError", 400))
    with pytest.raises(BackendFatalError):
        be.decide(make_request())


def test_api_unknown_exceptions_are_not_swallowed():
    be, _ = stub_backend(error=KeyError("programming error"))
    with pytest.raises(KeyError):
        be.decide(make_request())


def test_api_module_survives_without_the_sdk(monkeypatch):
    monkeypatch.setattr(api_module, "anthropic", None)
    be, _ = stub_backend(make_response())
    assert be.decide(make_request()).error is None
    with pytest.raises(BackendFatalError, match="not installed"):
        ApiBackend(BackendConfig())


# ---- file exchange -----------------------------------------------------------------


def fx_backend(tmp_path, digest="d1", timeout=5.0, poll=0.01):
    cfg = BackendConfig(kind="file", wait_timeout_s=timeout, poll_interval_s=poll)
    return FileExchangeBackend(cfg, tmp_path / "ex", digest, settle_s=0.0)


def test_exchange_writes_trigger_and_reads_decision(tmp_path):
    be = fx_backend(tmp_path)
    req = make_request(trigger_id=7, agent_index=2)
    trig, dec = be.paths_for(req)
    assert trig.name == "trigger_0007_a02.json" and dec.name == "decision_0007_a02.json"

    def external_agent():
        for _ in range(200):
            if trig.exists():
                payload = json.loads(trig.read_text(encoding="utf-8"))
                target = be.root / payload["write_decision_to"]
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_text('{"action": "HOLD", "sl_price": null, "tp_price": null}', encoding="utf-8")
                os.replace(tmp, target)
                return
            time.sleep(0.01)

    t = threading.Thread(target=external_agent)
    t.start()
    resp = be.decide(req)
    t.join()
    assert json.loads(resp.raw_text)["action"] == "HOLD"
    payload = json.loads(trig.read_text(encoding="utf-8"))
    assert payload["system_prompt"] == "SYSTEM" and payload["user_prompt"] == "USER"
    assert payload["write_decision_to"] == "decisions/decision_0007_a02.json"
    assert payload["decision_schema"]["required"]


def test_exchange_uses_existing_decision_without_waiting(tmp_path):
    be = fx_backend(tmp_path, timeout=0.05)
    req = make_request(trigger_id=1, agent_index=0)
    _, dec = be.paths_for(req)
    dec.write_text('{"action": "CLOSE"}', encoding="utf-8")
    assert be.decide(req).raw_text == '{"action": "CLOSE"}'


def test_exchange_times_out_with_a_resumable_message(tmp_path):
    ticks = iter(range(1000))
    cfg = BackendConfig(kind="file", wait_timeout_s=3.0, poll_interval_s=1.0)
    be = FileExchangeBackend(cfg, tmp_path / "ex", "d1", clock=lambda: float(next(ticks)), sleep=lambda s: None, settle_s=0.0)
    with pytest.raises(ExchangeTimeout, match="resume"):
        be.decide(make_request(trigger_id=3, agent_index=1))
    assert (be.triggers_dir / "trigger_0003_a01.json").exists()


def test_exchange_rejects_oversized_decision_files(tmp_path):
    be = fx_backend(tmp_path)
    req = make_request()
    _, dec = be.paths_for(req)
    dec.write_text("x" * (70 * 1024), encoding="utf-8")
    assert "too large" in be.decide(req).error


def test_exchange_retry_attempts_use_separate_files(tmp_path):
    be = fx_backend(tmp_path)
    first, _ = be.paths_for(make_request(attempt=0))
    retry, _ = be.paths_for(make_request(attempt=1))
    assert first != retry and retry.name.endswith("_r1.json")


def test_exchange_refuses_a_directory_from_another_run(tmp_path):
    fx_backend(tmp_path, digest="run-a")
    fx_backend(tmp_path, digest="run-a")  # same run reopens fine
    with pytest.raises(BackendFatalError, match="different run configuration"):
        fx_backend(tmp_path, digest="run-b")
