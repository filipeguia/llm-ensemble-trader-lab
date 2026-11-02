"""Execution backends: mock (offline), api (Anthropic SDK), file (external agents)"""
from __future__ import annotations

from pathlib import Path

from ..config import BackendConfig
from .api import ApiBackend
from .base import Backend, BackendFatalError, BackendResponse, DecisionRequest
from .file_exchange import ExchangeTimeout, FileExchangeBackend
from .mock import MockBackend


def make_backend(
    cfg: BackendConfig,
    *,
    kind: str | None = None,
    exchange_dir: str | Path | None = None,
    run_digest: str = "",
) -> Backend:
    kind = kind or cfg.kind
    if kind == "mock":
        return MockBackend(cfg)
    if kind == "api":
        return ApiBackend(cfg)
    if kind == "file":
        return FileExchangeBackend(cfg, exchange_dir or cfg.exchange_dir, run_digest)
    raise ValueError(f"unknown backend kind {kind!r}")


__all__ = [
    "ApiBackend",
    "Backend",
    "BackendFatalError",
    "BackendResponse",
    "DecisionRequest",
    "ExchangeTimeout",
    "FileExchangeBackend",
    "MockBackend",
    "make_backend",
]
