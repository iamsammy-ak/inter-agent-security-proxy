"""Runtime settings read from the environment (and an optional ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = (
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "google/gemma-4-31b-it:free",
)
DEFAULT_PORT = 8710


@dataclass(frozen=True)
class Settings:
    """Everything configurable. Secrets come only from the environment, never from files in the repo."""

    api_key: str | None = None
    models: tuple[str, ...] = DEFAULT_MODELS
    offline: bool = False
    secret: str | None = None
    audit_key: str | None = None
    policy_path: Path | None = None
    audit_path: Path | None = None
    classifier: str = "auto"

    @property
    def use_live_upstream(self) -> bool:
        return bool(self.api_key) and not self.offline


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_settings(*, use_dotenv: bool = True) -> Settings:
    """Build :class:`Settings`. ``AGENT_PROXY_OFFLINE=1`` forces the mock upstream even when a key is present."""
    if use_dotenv:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
            pass
    models = tuple(m.strip() for m in os.environ.get("OPENROUTER_MODELS", "").split(",") if m.strip())
    policy = os.environ.get("AGENT_PROXY_POLICY")
    audit = os.environ.get("AGENT_PROXY_AUDIT_LOG")
    return Settings(
        api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        models=models or DEFAULT_MODELS,
        offline=_flag(os.environ.get("AGENT_PROXY_OFFLINE")),
        secret=os.environ.get("AGENT_PROXY_SECRET") or None,
        audit_key=os.environ.get("AGENT_PROXY_AUDIT_KEY") or None,
        policy_path=Path(policy) if policy else None,
        audit_path=Path(audit) if audit else None,
        classifier=os.environ.get("AGENT_PROXY_CLASSIFIER", "auto"),
    )
