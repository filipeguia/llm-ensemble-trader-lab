"""Direct API backend using the Anthropic Python SDK

The key is read only from the ANTHROPIC_API_KEY environment variable. No
sampling parameters are sent (newer models reject non-default values), so
diversity between ensemble members comes from the model's default sampling.

Failure handling:
- authentication, permission, bad request and not found are fatal: retrying
  cannot help and the run stops with a clear message
- rate limit, server and connection errors are retried by the SDK itself
  (max_retries), and if they still fail the member abstains for that trigger
- refusals and answers cut at max_tokens abstain as well
"""
from __future__ import annotations

import os
from typing import Any

from ..config import BackendConfig
from ..cost import Usage
from ..schema import DECISION_JSON_SCHEMA
from .base import BackendFatalError, BackendResponse, DecisionRequest

try:  # the SDK is an optional dependency
    import anthropic
except ImportError:  # pragma: no cover - exercised only when the SDK is absent
    anthropic = None  # type: ignore[assignment]

API_KEY_ENV = "ANTHROPIC_API_KEY"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_FATAL = ("AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError")
_TRANSIENT = ("RateLimitError", "APIStatusError", "APIConnectionError")


def _is_one_of(exc: BaseException, names: tuple[str, ...]) -> bool:
    if anthropic is None:
        return False
    return any(isinstance(exc, getattr(anthropic, n)) for n in names if hasattr(anthropic, n))


def _usage_from(response: Any) -> Usage:
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )


class ApiBackend:
    name = "api"

    def __init__(self, cfg: BackendConfig, client: Any | None = None):
        self.cfg = cfg
        self.client = client if client is not None else self._make_client(cfg)

    @staticmethod
    def _make_client(cfg: BackendConfig) -> Any:
        if anthropic is None:
            raise BackendFatalError("the 'anthropic' package is not installed, run: pip install anthropic")
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise BackendFatalError(f"{API_KEY_ENV} is not set")
        return anthropic.Anthropic(
            api_key=key, timeout=cfg.request_timeout_s, max_retries=cfg.sdk_max_retries
        )

    def build_kwargs(self, request: DecisionRequest) -> dict[str, Any]:
        cfg = self.cfg
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            # The system prompt is identical for every call, mark it cacheable. Prompts
            # shorter than the model's minimum cacheable length are simply not cached
            "system": [
                {"type": "text", "text": request.system_prompt, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": request.user_prompt}],
        }
        output_config: dict[str, Any] = {}
        if cfg.effort:
            output_config["effort"] = cfg.effort
        if cfg.structured_output:
            output_config["format"] = {"type": "json_schema", "schema": DECISION_JSON_SCHEMA}
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    def decide(self, request: DecisionRequest) -> BackendResponse:
        kwargs = self.build_kwargs(request)
        try:
            if self.cfg.server_side_fallback:
                response = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                response = self.client.messages.create(**kwargs)
        except BackendFatalError:
            raise
        except Exception as exc:
            if _is_one_of(exc, _FATAL):
                raise BackendFatalError(f"{type(exc).__name__}: {exc}") from exc
            if _is_one_of(exc, _TRANSIENT):
                return BackendResponse(error=f"{type(exc).__name__}: {exc}")
            raise

        usage = _usage_from(response)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            return BackendResponse(usage=usage, error=f"refusal (category={category})")
        if stop_reason == "max_tokens":
            return BackendResponse(usage=usage, error="answer truncated at max_tokens")
        # Read blocks by type, thinking blocks may come first
        text = "".join(
            getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            return BackendResponse(usage=usage, error="empty answer")
        return BackendResponse(raw_text=text, usage=usage)
