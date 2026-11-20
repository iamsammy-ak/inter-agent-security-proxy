"""Upstream chat-completion providers behind the proxy.

``OpenRouterUpstream`` forwards OpenAI-format requests to OpenRouter, trying each configured model in turn and
caching successful replies on disk. ``MockUpstream`` is a deterministic offline stand-in used by the tests, the
demo and whenever no ``OPENROUTER_API_KEY`` is set. The mock is deliberately *gullible*: it obeys any
``tool_name(arg="value")`` call it reads in the latest message, which models the worst case and shows what the
policy layer catches when a model does fall for an injection.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .injection.unicode import DATAMARK

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CACHE_DIR = Path(".cache") / "llm"


class UpstreamError(RuntimeError):
    """No upstream model produced an answer."""


class Upstream(Protocol):
    """Anything that turns an OpenAI chat-completions request body into a response body."""

    name: str

    def create(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def message_text(message: dict[str, Any]) -> str:
    """Plain text of an OpenAI message whose content is a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def complete(upstream: Upstream, messages: list[dict[str, Any]], **params: Any) -> str:
    """Convenience wrapper: send ``messages`` and return the first choice's text."""
    response = upstream.create({"messages": messages, **params})
    return message_text(response["choices"][0]["message"])


_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(([^()]*)\)")
_KWARG = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,\s)]+)")


def _parse_value(raw: str) -> Any:
    if raw[:1] in "\"'":
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


class MockUpstream:
    """Deterministic, gullible offline model (a stand-in, not a language model)."""

    name = "mock/gullible-agent"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = list(payload.get("messages") or [])
        tools = {
            str(t["function"]["name"])
            for t in payload.get("tools") or []
            if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict)
        }
        latest: dict[str, Any] = next((m for m in reversed(messages) if m.get("role") in ("user", "tool")), {})
        text = message_text(latest).replace(DATAMARK, " ")
        calls: list[dict[str, Any]] = []
        for match in _CALL.finditer(text):
            if match.group(1) not in tools:
                continue
            args = {k: _parse_value(v) for k, v in _KWARG.findall(match.group(2))}
            call_id = "call_" + hashlib.sha256(f"{len(calls)}{match.group()}".encode()).hexdigest()[:10]
            calls.append({"id": call_id, "type": "function",
                          "function": {"name": match.group(1), "arguments": json.dumps(args)}})
        if calls:
            message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": calls}
            finish = "tool_calls"
        else:
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            message = {"role": "assistant",
                       "content": f"[mock] Read {len(text)} characters. First line: {first_line[:100]}"}
            finish = "stop"
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
        prompt_tokens = sum(len(message_text(m).split()) for m in messages)
        return {
            "id": f"mock-{digest}",
            "object": "chat.completion",
            "created": 0,
            "model": self.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
        }


class OpenRouterUpstream:
    """OpenRouter through the OpenAI SDK, with per-model fallback and an on-disk cache."""

    name = "openrouter"

    def __init__(
        self, settings: Settings, *, cache_dir: Path | None = DEFAULT_CACHE_DIR, timeout: float = 60.0
    ) -> None:
        if not settings.api_key:
            raise UpstreamError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI

        self.settings = settings
        self.cache_dir = cache_dir
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.api_key, timeout=timeout, max_retries=0,
                              default_headers={"X-Title": "Inter-Agent Security Proxy"})
        self.last_errors: list[str] = []

    def _cache_file(self, body: dict[str, Any], models: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps({"models": models, "body": body}, sort_keys=True, default=str)
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:40]}.json"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        body = {k: v for k, v in payload.items() if k not in ("model", "stream")}
        requested = payload.get("model")
        models = [str(requested)] if requested and requested not in ("auto", "mock") else []
        models += [m for m in self.settings.models if m not in models]
        path = self._cache_file(body, models)
        if path and path.exists():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return cached
        self.last_errors = []
        for model in models:
            try:
                response = self._client.chat.completions.create(model=model, **body)
            except RateLimitError:
                self.last_errors.append(f"{model}: busy (HTTP 429)")
                continue
            except APIConnectionError as exc:  # includes timeouts
                self.last_errors.append(f"{model}: {type(exc).__name__}")
                continue
            except APIStatusError as exc:
                if exc.status_code == 401:
                    raise UpstreamError("OpenRouter rejected the API key (HTTP 401)") from exc
                self.last_errors.append(f"{model}: HTTP {exc.status_code}")
                continue
            result: dict[str, Any] = response.model_dump(exclude_none=True)
            if result.get("choices"):
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result), encoding="utf-8")
                return result
            self.last_errors.append(f"{model}: empty reply")
        raise UpstreamError("no upstream model answered: " + "; ".join(self.last_errors[-6:]))


def select_upstream(settings: Settings, *, offline: bool = False) -> Upstream:
    """OpenRouter when a key is configured and offline mode is off; otherwise the mock."""
    if settings.use_live_upstream and not offline:
        return OpenRouterUpstream(settings)
    return MockUpstream()
