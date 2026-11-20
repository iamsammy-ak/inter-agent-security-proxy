from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis.configuration import set_hypothesis_home_dir

from agent_proxy.audit import AuditLog
from agent_proxy.demo import demo_policy_path
from agent_proxy.injection.classifier import InjectionClassifier
from agent_proxy.pii.names import spacy_model_available
from agent_proxy.pii.vault import TokenVault
from agent_proxy.policy import PolicyFile, load_policy
from agent_proxy.proxy import SecurityProxy

# Keep Hypothesis's example database and constant cache out of the working tree.
set_hypothesis_home_dir(Path(tempfile.gettempdir()) / "agent-proxy-hypothesis")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call a live model, even when a key is configured on the machine."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_PROXY_OFFLINE", "1")
    monkeypatch.setenv("AGENT_PROXY_CLASSIFIER", "hashing")


def bge_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    cache = os.environ.get("FASTEMBED_CACHE_PATH")
    return bool(cache) and any("bge-small-en-v1.5" in p.name for p in Path(cache).glob("*"))


AVAILABLE = {"bge": bge_available, "spacy": spacy_model_available}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker in item.iter_markers("models"):
            needed = marker.args[0] if marker.args else "bge"
            if not AVAILABLE[needed]():
                item.add_marker(pytest.mark.skip(reason=f"{needed} model not available"))


@pytest.fixture(scope="session")
def hashing_classifier() -> InjectionClassifier:
    return InjectionClassifier.load("hashing")


@pytest.fixture()
def vault() -> TokenVault:
    return TokenVault(b"test-secret")


@pytest.fixture()
def policy() -> PolicyFile:
    return load_policy(demo_policy_path())


@pytest.fixture()
def proxy(policy: PolicyFile, hashing_classifier: InjectionClassifier) -> SecurityProxy:
    return SecurityProxy(policy, vault=TokenVault(b"test-secret"), classifier=hashing_classifier,
                         audit=AuditLog(key=b"audit-key"))
