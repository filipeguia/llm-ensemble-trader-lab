"""End-to-end checks on synthetic data with the mock backend: no key, no network"""
from __future__ import annotations

import csv
import json
import os
import threading
import time

import pytest

from llm_ensemble_lab.backends import DecisionRequest, MockBackend
from llm_ensemble_lab.backends.base import BackendResponse
from llm_ensemble_lab.cli import main
from llm_ensemble_lab.config import ConfigError, config_from_dict
from llm_ensemble_lab.cost import Usage
from llm_ensemble_lab.ensemble_compare import compare_sub_ensembles, load_vote_records
from llm_ensemble_lab.results import aggregate_summaries, load_summary
from llm_ensemble_lab.runner import CostCapError, estimate_cost, prepare, run_experiment
from llm_ensemble_lab.voting import ConsensusRule


def quiet(_msg: str) -> None:
    pass


def test_dry_run_writes_all_outputs(small_cfg, tmp_path):
    out = tmp_path / "run"
    summary = run_experiment(small_cfg, out, dry_run=True, echo=quiet)
    for name in ("config.json", "decisions.jsonl", "trades.csv", "summary.json", "cost.json"):
        assert (out / name).exists(), name
    assert summary["completed"] and summary["dry_run"] and summary["backend"] == "mock"
    assert summary["n_triggers_consulted"] > 5
    assert summary["n_agent_answers"] == summary["n_triggers_consulted"] * small_cfg.ensemble.n_agents
    assert summary["spent_usd"] == 0.0
    lines = (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["n_triggers_consulted"]
    first = json.loads(lines[0])
    assert len(first["members"]) == small_cfg.ensemble.n_agents
    assert {"action", "reason", "counts"} <= set(first["consensus"])
    with open(out / "trades.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == summary["n_trades"]


def test_dry_run_is_reproducible(small_cfg, tmp_path):
    a = run_experiment(small_cfg, tmp_path / "a", dry_run=True, echo=quiet)
    b = run_experiment(small_cfg, tmp_path / "b", dry_run=True, echo=quiet)
    assert a == b
    assert (tmp_path / "a" / "decisions.jsonl").read_text() == (tmp_path / "b" / "decisions.jsonl").read_text()
    assert (tmp_path / "a" / "trades.csv").read_text() == (tmp_path / "b" / "trades.csv").read_text()


def test_dry_run_replaces_a_real_backend_without_touching_the_network(small_cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = config_from_dict({**small_cfg.to_dict(), "backend": {**small_cfg.to_dict()["backend"], "kind": "api"}})
    messages = []
    summary = run_experiment(cfg, tmp_path / "r", dry_run=True, echo=messages.append)
    assert summary["backend"] == "mock"
    assert any("replaced by the deterministic mock" in m for m in messages)


def test_different_mock_seeds_change_the_votes(small_cfg, tmp_path):
    other = config_from_dict({**small_cfg.to_dict(), "backend": {**small_cfg.to_dict()["backend"], "mock_seed": 5}})
    run_experiment(small_cfg, tmp_path / "a", dry_run=True, echo=quiet)
    run_experiment(other, tmp_path / "b", dry_run=True, echo=quiet)
    assert (tmp_path / "a" / "decisions.jsonl").read_text() != (tmp_path / "b" / "decisions.jsonl").read_text()


def test_existing_output_directory_needs_force(small_cfg, tmp_path):
    out = tmp_path / "run"
    run_experiment(small_cfg, out, dry_run=True, echo=quiet)
    with pytest.raises(FileExistsError):
        run_experiment(small_cfg, out, dry_run=True, echo=quiet)
    run_experiment(small_cfg, out, dry_run=True, force=True, echo=quiet)


def test_estimate_matches_the_number_of_calls_actually_made(small_cfg, tmp_path):
    prepared = prepare(small_cfg)
    est = estimate_cost(small_cfg, prepared)
    summary = run_experiment(small_cfg, tmp_path / "r", dry_run=True, echo=quiet)
    # a trigger on the last bar is skipped by both the estimator and the loop, and with
    # parse_retries=0 every agent answer costs exactly one backend call
    assert est.n_triggers == summary["n_triggers_consulted"]
    assert est.n_calls == est.n_triggers * small_cfg.ensemble.n_agents == summary["n_backend_calls"]


def test_trigger_free_data_is_a_clear_error(small_cfg, tmp_path):
    cfg = config_from_dict({**small_cfg.to_dict(), "trigger": {**small_cfg.to_dict()["trigger"], "z_threshold": 500.0}})
    with pytest.raises(ConfigError, match="no events"):
        run_experiment(cfg, tmp_path / "r", dry_run=True, echo=quiet)


# ---- cost cap and budget with a stubbed API backend --------------------------------


class ExpensiveBackend:
    name = "expensive"

    def __init__(self):
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        raw = json.dumps({"action": "HOLD", "sl_price": None, "tp_price": None, "reasoning": "x"})
        # about $1 per call at the table price of the default model
        return BackendResponse(raw_text=raw, usage=Usage(input_tokens=500_000, output_tokens=0))


def api_cfg(small_cfg, **cost):
    data = small_cfg.to_dict()
    data["backend"]["kind"] = "api"
    data["cost"].update(cost)
    return config_from_dict(data)


def test_expected_cost_above_the_cap_refuses_before_any_call(small_cfg, tmp_path):
    cfg = api_cfg(small_cfg, max_usd=0.0001)
    backend = ExpensiveBackend()
    with pytest.raises(CostCapError, match="above the cap"):
        run_experiment(cfg, tmp_path / "r", backend=backend, echo=quiet)
    assert backend.calls == 0


def test_runtime_budget_stops_the_run_and_keeps_partial_results(small_cfg, tmp_path):
    # the pre-run estimate passes the cap, but the stub reports a huge usage per call
    cfg = api_cfg(small_cfg, max_usd=2.0)
    backend = ExpensiveBackend()
    summary = run_experiment(cfg, tmp_path / "r", backend=backend, echo=quiet)
    assert summary["completed"] is False
    assert "BudgetExceeded" in summary["aborted_reason"]
    assert backend.calls == 3
    assert (tmp_path / "r" / "summary.json").exists()


def test_api_run_without_a_key_fails_cleanly(small_cfg, tmp_path, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = api_cfg(small_cfg, max_usd=100.0)
    code = main(["run", "--out", str(tmp_path / "r"), "--backend", "api", "--set", "data.synthetic.n_bars=300"])
    assert code == 2
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        run_experiment(cfg, tmp_path / "r2", echo=quiet)


# ---- file exchange through the runner, with simulated external agents ---------------


def start_agents(root, n_agents, backend_cfg, stop):
    """Threads that behave like external agent CLIs: read trigger files, write decision files"""
    mock = MockBackend(backend_cfg)

    def worker(idx):
        while not stop.is_set():
            for path in sorted((root / "triggers").glob(f"trigger_*_a{idx:02d}.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                target = root / payload["write_decision_to"]
                if target.exists():
                    continue
                req = DecisionRequest(
                    payload["trigger_id"], payload["agent_index"], 0,
                    payload["system_prompt"], payload["user_prompt"], payload["context"],
                )
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_text(mock.decide(req).raw_text, encoding="utf-8")
                os.replace(tmp, target)
            time.sleep(0.005)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_agents)]
    for t in threads:
        t.start()
    return threads


def test_file_backend_end_to_end_matches_the_mock_and_resumes(small_cfg, tmp_path):
    data = small_cfg.to_dict()
    data["data"]["synthetic"]["n_bars"] = 300
    data["ensemble"]["n_agents"] = 3
    data["backend"].update(
        kind="file", exchange_dir=str(tmp_path / "ex"), poll_interval_s=0.005, wait_timeout_s=60.0, settle_s=0.0
    )
    cfg = config_from_dict(data)

    stop = threading.Event()
    threads = start_agents(tmp_path / "ex", 3, cfg.backend, stop)
    try:
        file_summary = run_experiment(cfg, tmp_path / "file", echo=quiet)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
    assert file_summary["backend"] == "file" and file_summary["completed"]

    mock_data = {**data, "backend": {**data["backend"], "kind": "mock"}}
    run_experiment(config_from_dict(mock_data), tmp_path / "mock", echo=quiet)
    file_lines = (tmp_path / "file" / "decisions.jsonl").read_text()
    assert file_lines == (tmp_path / "mock" / "decisions.jsonl").read_text()

    # Resume: no agents are running now, every decision file already exists
    resumed = config_from_dict({**data, "backend": {**data["backend"], "wait_timeout_s": 2.0}})
    run_experiment(resumed, tmp_path / "resumed", echo=quiet)
    assert (tmp_path / "resumed" / "decisions.jsonl").read_text() == file_lines


def test_file_backend_without_agents_times_out_and_keeps_partial_results(small_cfg, tmp_path):
    data = small_cfg.to_dict()
    data["data"]["synthetic"]["n_bars"] = 300
    data["backend"].update(
        kind="file", exchange_dir=str(tmp_path / "ex"), poll_interval_s=0.01, wait_timeout_s=0.1, settle_s=0.0
    )
    summary = run_experiment(config_from_dict(data), tmp_path / "r", echo=quiet)
    assert summary["completed"] is False and "ExchangeTimeout" in summary["aborted_reason"]
    assert summary["n_triggers_consulted"] == 0
    assert any((tmp_path / "ex" / "triggers").glob("trigger_0001_a*.json"))


# ---- ensemble comparison -----------------------------------------------------------


def test_compare_n4_against_n8(small_cfg, tmp_path):
    cfg = config_from_dict({**small_cfg.to_dict(), "ensemble": {**small_cfg.to_dict()["ensemble"], "n_agents": 8}})
    out = tmp_path / "eight"
    run_experiment(cfg, out, dry_run=True, echo=quiet)
    records = load_vote_records(out / "decisions.jsonl")
    rule = ConsensusRule.from_config(cfg.ensemble.rule)
    res = compare_sub_ensembles(records, rule, 4)
    assert res["n_agents"] == 8 and res["sub_size"] == 4
    assert res["required_full"] == 5 and res["required_sub"] == 3
    assert res["n_subsets_total"] == 70 and res["n_subsets_used"] == 70 and not res["subsets_sampled"]
    agree = res["sub_agreement_with_reference"]
    assert 0.0 <= agree["min"] <= agree["mean"] <= agree["max"] <= 1.0
    assert 0 <= res["triggers_with_any_flip"] <= res["n_triggers"]


def test_compare_samples_when_there_are_too_many_subsets(small_cfg, tmp_path):
    cfg = config_from_dict({**small_cfg.to_dict(), "ensemble": {**small_cfg.to_dict()["ensemble"], "n_agents": 12}})
    run_experiment(cfg, tmp_path / "r", dry_run=True, echo=quiet)
    records = load_vote_records(tmp_path / "r" / "decisions.jsonl")
    rule = ConsensusRule.from_config(cfg.ensemble.rule)
    res = compare_sub_ensembles(records, rule, 6, max_combos=40, seed=1)
    assert res["subsets_sampled"] and res["n_subsets_used"] == 40
    assert res == compare_sub_ensembles(records, rule, 6, max_combos=40, seed=1)


def test_comparing_the_full_size_is_rejected(small_cfg, tmp_path):
    run_experiment(small_cfg, tmp_path / "r", dry_run=True, echo=quiet)
    records = load_vote_records(tmp_path / "r" / "decisions.jsonl")
    with pytest.raises(ValueError):
        compare_sub_ensembles(records, ConsensusRule(), small_cfg.ensemble.n_agents)


def test_absolute_quorum_keeps_its_share_in_the_sub_ensembles():
    from llm_ensemble_lab.ensemble_compare import VoteRecord
    from llm_ensemble_lab.schema import Decision

    buy = Decision("BUY", 99.0, 101.0)
    records = [VoteRecord(i, tuple([buy] * 8)) for i in range(3)]
    res = compare_sub_ensembles(records, ConsensusRule(mode="quorum", quorum_votes=6), 4)
    assert res["required_full"] == 6 and res["required_sub"] == 3


def test_unanimous_agents_make_every_subset_agree():
    from llm_ensemble_lab.ensemble_compare import VoteRecord
    from llm_ensemble_lab.schema import Decision

    buy = Decision("BUY", 99.0, 101.0)
    records = [VoteRecord(i, tuple([buy] * 6)) for i in range(4)]
    res = compare_sub_ensembles(records, ConsensusRule(mode="majority"), 3)
    assert res["sub_agreement_with_reference"]["min"] == 1.0
    assert res["triggers_with_any_flip"] == 0


# ---- aggregation -------------------------------------------------------------------


def test_aggregate_across_runs(small_cfg, tmp_path):
    dirs = []
    for seed in (0, 1, 2):
        cfg = config_from_dict({**small_cfg.to_dict(), "backend": {**small_cfg.to_dict()["backend"], "mock_seed": seed}})
        out = tmp_path / f"s{seed}"
        run_experiment(cfg, out, dry_run=True, echo=quiet)
        dirs.append(out)
    agg = aggregate_summaries([load_summary(d) for d in dirs])
    assert agg["n_runs"] == 3
    m = agg["metrics"]["n_trades"]
    assert m["min"] <= m["median"] <= m["max"] and m["std"] >= 0
    assert "completed" not in agg["metrics"]  # booleans are not metrics
    with pytest.raises(ValueError):
        aggregate_summaries([])


# ---- CLI ---------------------------------------------------------------------------


def test_cli_dry_run_compare_and_aggregate(tmp_path, capsys):
    out = tmp_path / "cli"
    args = ["--set", "data.synthetic.n_bars=500"]
    assert main(["estimate", *args]) == 0
    assert main(["run", "--dry-run", "--agents", "8", "--out", str(out), *args]) == 0
    assert main(["compare", str(out), "--sub-size", "4"]) == 0
    assert (out / "ensemble_compare.json").exists()
    assert main(["compare", str(out), "--sub-size", "4", "--fractions", "0.5", "0.7"]) == 0
    assert main(["aggregate", str(out), "--out", str(tmp_path / "agg.json")]) == 0
    assert (tmp_path / "agg.json").exists()
    text = capsys.readouterr().out
    assert "cost estimate" in text and "agreement with full ensemble" in text


def test_cli_generate_and_csv_input(tmp_path, capsys):
    csv_path = tmp_path / "bars.csv"
    assert main(["generate", "--out", str(csv_path), "--bars", "600", "--seed", "3"]) == 0
    assert main(["run", "--dry-run", "--csv", str(csv_path), "--out", str(tmp_path / "r")]) == 0
    assert "results in" in capsys.readouterr().out


def test_cli_reports_errors_with_exit_code_2(tmp_path, capsys):
    assert main(["run", "--dry-run", "--out", str(tmp_path / "r"), "--set", "trigger.nope=1"]) == 2
    assert main(["run", "--dry-run", "--out", str(tmp_path / "r"), "--set", "ensemble.n_agents=0"]) == 2
    assert main(["compare", str(tmp_path / "missing")]) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 3


def test_cli_set_parses_json_values(tmp_path):
    out = tmp_path / "r"
    code = main(
        [
            "run", "--dry-run", "--out", str(out),
            "--set", "data.synthetic.n_bars=400",
            "--set", "ensemble.rule.mode=\"majority\"",
            "--set", "ensemble.rule.priority=[\"BUY\",\"SELL\",\"HOLD\",\"CLOSE\"]",
        ]
    )
    assert code == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["ensemble"]["rule"]["mode"] == "majority"
    assert cfg["ensemble"]["rule"]["priority"][0] == "BUY"
