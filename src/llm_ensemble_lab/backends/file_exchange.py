"""File-exchange backend for agent CLIs

Instead of calling an API, the runner writes one trigger file per (trigger,
agent) and waits until an external agent writes the matching decision file.
The external agent can be any process that reads JSON and writes JSON, for
example a command line coding agent following the prompt templates in agents/

Layout under the exchange directory:

    exchange_meta.json                          run fingerprint
    triggers/trigger_0007_a02.json              written by the runner
    decisions/decision_0007_a02.json            written by the external agent

Decision files that already exist are used without waiting, so a run that was
interrupted (or timed out) resumes where it stopped when started again with
the same configuration
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from ..config import BackendConfig
from ..schema import DECISION_JSON_SCHEMA
from .base import BackendFatalError, BackendResponse, DecisionRequest

MAX_DECISION_BYTES = 64 * 1024


class ExchangeTimeout(BackendFatalError):
    """No decision file appeared within the wait timeout"""


class FileExchangeBackend:
    name = "file"

    def __init__(
        self,
        cfg: BackendConfig,
        exchange_dir: str | Path,
        run_digest: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        settle_s: float | None = None,
    ):
        self.cfg = cfg
        self.root = Path(exchange_dir)
        self.run_digest = run_digest
        self.triggers_dir = self.root / "triggers"
        self.decisions_dir = self.root / "decisions"
        self._clock = clock
        self._sleep = sleep
        self._settle_s = cfg.settle_s if settle_s is None else settle_s
        self.triggers_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self._check_fingerprint()

    def _check_fingerprint(self) -> None:
        meta = self.root / "exchange_meta.json"
        if meta.exists():
            try:
                old = json.loads(meta.read_text(encoding="utf-8")).get("run_digest")
            except (OSError, json.JSONDecodeError):
                old = None
            if old != self.run_digest:
                raise BackendFatalError(
                    "the exchange directory belongs to a different run configuration, "
                    "use a new exchange directory so stale decisions are not replayed"
                )
        else:
            self._atomic_write(meta, json.dumps({"run_digest": self.run_digest}, indent=2))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def paths_for(self, request: DecisionRequest) -> tuple[Path, Path]:
        # Names are built from integers only, so nothing from the context reaches a path
        suffix = f"_r{request.attempt}" if request.attempt else ""
        stem = f"{request.trigger_id:04d}_a{request.agent_index:02d}{suffix}"
        return (
            self.triggers_dir / f"trigger_{stem}.json",
            self.decisions_dir / f"decision_{stem}.json",
        )

    def _write_trigger(self, trigger_path: Path, decision_path: Path, request: DecisionRequest) -> None:
        payload: dict[str, Any] = {
            "format_version": 1,
            "run_digest": self.run_digest,
            "trigger_id": request.trigger_id,
            "agent_index": request.agent_index,
            "write_decision_to": decision_path.relative_to(self.root).as_posix(),
            "decision_schema": DECISION_JSON_SCHEMA,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "context": request.context,
        }
        self._atomic_write(trigger_path, json.dumps(payload, indent=2))

    def _is_ready(self, path: Path) -> bool:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age >= self._settle_s

    def decide(self, request: DecisionRequest) -> BackendResponse:
        trigger_path, decision_path = self.paths_for(request)
        if not decision_path.exists():
            self._write_trigger(trigger_path, decision_path, request)
        deadline = self._clock() + self.cfg.wait_timeout_s
        while not self._is_ready(decision_path):
            if self._clock() >= deadline:
                raise ExchangeTimeout(
                    f"no decision after {self.cfg.wait_timeout_s:.0f}s, waiting for "
                    f"{decision_path.relative_to(self.root).as_posix()}, "
                    "start the run again with the same configuration to resume"
                )
            self._sleep(self.cfg.poll_interval_s)
        size = decision_path.stat().st_size
        if size > MAX_DECISION_BYTES:
            return BackendResponse(error=f"decision file too large ({size} bytes)")
        return BackendResponse(raw_text=decision_path.read_text(encoding="utf-8", errors="replace"))
