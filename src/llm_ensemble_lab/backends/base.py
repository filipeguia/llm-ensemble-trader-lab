"""Backend contract shared by the mock, API and file-exchange implementations"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..cost import Usage


class BackendFatalError(RuntimeError):
    """The backend cannot continue (bad credentials, bad request, exchange timeout)

    Transient failures do not raise this, they come back as a BackendResponse
    with `error` set so the ensemble can treat the member as abstaining
    """


@dataclass(frozen=True)
class DecisionRequest:
    trigger_id: int
    agent_index: int
    attempt: int
    system_prompt: str
    user_prompt: str
    context: dict[str, Any]


@dataclass
class BackendResponse:
    raw_text: str = ""
    usage: Usage = field(default_factory=Usage)
    # Set for transport-level failures only, content problems are found by validation
    error: str | None = None


class Backend(Protocol):
    name: str

    def decide(self, request: DecisionRequest) -> BackendResponse: ...
