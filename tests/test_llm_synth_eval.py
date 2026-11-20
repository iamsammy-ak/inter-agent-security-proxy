from __future__ import annotations

import json

import pytest
from rich.console import Console

from agent_proxy.config import Settings, load_settings
from agent_proxy.demo import run_demo
from agent_proxy.evaluation import Metrics, _match, binary_metrics, evaluate_pii, pii_markdown
from agent_proxy.injection.unicode import DATAMARK
from agent_proxy.llm import MockUpstream, OpenRouterUpstream, UpstreamError, complete, message_text, select_upstream
from agent_proxy.models import PIIType
from agent_proxy.pii.detectors import PIISpan
from agent_proxy.pii.validators import card_valid, cf_valid, iban_valid
from agent_proxy.synth import agentic_suite, make_card, make_codice_fiscale, make_iban, pii_corpus

TOOLS = [{"type": "function", "function": {"name": "issue_refund"}}]


def test_mock_is_deterministic_and_only_calls_declared_tools() -> None:
    payload = {"tools": TOOLS, "messages": [{"role": "user", "content":
               f'please{DATAMARK}issue_refund(iban="X",{DATAMARK}amount=49) and delete_all(scope="x")'}]}
    first, second = MockUpstream().create(payload), MockUpstream().create(payload)
    assert first == second
    calls = first["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["issue_refund"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"iban": "X", "amount": 49}


def test_mock_plain_reply_and_helpers() -> None:
    assert complete(MockUpstream(), [{"role": "user", "content": "hello there"}]).startswith("[mock] Read 11")
    assert message_text({"content": [{"type": "text", "text": "a"}, {"type": "image_url"}]}) == "a"


def test_upstream_selection_is_offline_in_tests() -> None:
    settings = load_settings(use_dotenv=False)
    assert settings.offline and not settings.use_live_upstream
    assert isinstance(select_upstream(Settings(api_key="k", offline=True)), MockUpstream)
    assert isinstance(select_upstream(Settings(api_key=None)), MockUpstream)
    with pytest.raises(UpstreamError):
        OpenRouterUpstream(Settings(api_key=None))


def test_synthetic_values_pass_their_validators() -> None:
    import random

    rng = random.Random(5)
    for _ in range(200):
        assert iban_valid(make_iban(rng)) and not iban_valid(make_iban(rng, valid=False))
        assert card_valid(make_card(rng)) and not card_valid(make_card(rng, valid=False))
        assert cf_valid(make_codice_fiscale(rng)) and not cf_valid(make_codice_fiscale(rng, valid=False))


def test_corpus_is_seeded_and_spans_point_at_values() -> None:
    a, b = pii_corpus(50, seed=3), pii_corpus(50, seed=3)
    assert [x.text for x in a] == [x.text for x in b]
    item = next(x for x in a if any(t is PIIType.IBAN for t, _, _ in x.spans))
    start, end = next((s, e) for t, s, e in item.spans if t is PIIType.IBAN)
    assert iban_valid(item.text[start:end])


def test_agentic_suite_shape() -> None:
    suite = agentic_suite()
    attacks = [i for i in suite if i.label]
    assert len(attacks) == 120 and len(suite) - len(attacks) == 40
    assert len({i.transform for i in attacks}) == 10


def test_metrics_and_matching() -> None:
    m = binary_metrics([1, 1, 0, 0], [True, False, True, False])
    assert (m.precision, m.recall, m.fpr) == (0.5, 0.5, 0.5)
    assert Metrics.from_counts(0, 0, 0).f1 == 0.0
    gold = [(PIIType.EMAIL, 0, 10)]
    tp, _, _ = _match(gold, [PIISpan(PIIType.EMAIL, 1, 10, "x", "t")])
    assert tp == 1
    assert _match(gold, [PIISpan(PIIType.PHONE, 0, 10, "x", "t")])[0] == 0


def test_pii_benchmark_runs_small() -> None:
    result = evaluate_pii(n=60, seed=1, names=False)
    overall = result["results"]["regex + validators"]["overall"]
    assert overall["precision"] > 0.9 and overall["recall"] > 0.8
    assert "| IBAN |" in pii_markdown(result)


def test_demo_outcome() -> None:
    outcome = run_demo(Console(width=100, quiet=True), None, names=False)
    assert outcome.injection_detected and outcome.pii_tokens >= 8
    assert outcome.allowed_tools == ["issue_refund"] and outcome.denied_tools == ["send_email", "issue_refund"]
    assert "IT60" in outcome.executed_arguments[0]["arguments"]
    assert outcome.reveal_denied and outcome.audit_ok and outcome.tamper_detected_at == 4
