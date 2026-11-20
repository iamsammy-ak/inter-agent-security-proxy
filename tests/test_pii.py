from __future__ import annotations

import random
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_proxy.models import PermissionDenied, PIIType, Principal
from agent_proxy.pii.detectors import TOKEN_RE, PIIDetector
from agent_proxy.pii.masker import Masker
from agent_proxy.pii.names import SpacyNameFinder
from agent_proxy.pii.vault import TokenVault, canonical
from agent_proxy.synth import make_secret

DETECTOR = PIIDetector()


def types(text: str) -> list[str]:
    return [s.type.value for s in DETECTOR.detect(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Write to giulia.rossi@example.com today", "EMAIL"),
        ("Cell +39 347 123 4567", "PHONE"),
        ("Chiamare il 347 123 4567", "PHONE"),
        ("Ufficio 02 1234 5678", "PHONE"),
        ("tel 3471234567", "PHONE"),
        ("London office +44 20 7946 0958", "PHONE"),
        ("US line (415) 555-2671", "PHONE"),
        ("IBAN IT60 X054 2811 1010 0000 0123 456 BIC BPMOIT22", "IBAN"),
        ("Card 4111-1111-1111-1111 exp 12/27", "CARD"),
        ("CF RSSMRA85T10A562S", "CODICE_FISCALE"),
        ("from 192.168.1.20 at noon", "IP"),
        ("v6 2001:db8::8a2e:370:7334 seen", "IP"),
        ("born on 12/03/1985", "DOB"),
        ("nata il 12 marzo 1985", "DOB"),
        ("DOB: 1985-03-12", "DOB"),
        ('api_key="Zx9kLm2Qp7Rt4Vw8"', "SECRET"),
    ],
)
def test_each_class_is_detected(text: str, expected: str) -> None:
    assert types(text) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Order 4111 1111 1111 1112 shipped",  # fails Luhn
        "Reference IT61X0542811101000000123456",  # wrong IBAN check digits
        "Code RSSMRA85T10A562A",  # wrong codice fiscale control character
        "Invoice date 12/03/2024, meeting 2024-05-01",  # dates without a birth context
        "Release 2.4.1 at 10:30:00, MAC 00:1A:2B:3C:4D:5E",
        "Commit 9f2c1e7b4a6d8e0f1a2b3c4d5e6f708192a3b4c5 and id 123e4567-e89b-12d3-a456-426614174000",
        "Invoice n. 0012345678 totals 1.234,56 EUR",
        "password: hunter",  # too short and low entropy
        "Reference FR18 3166 5528 7196 4328 6872 471 failed",  # invalid IBAN: its digits are not a card
    ],
)
def test_hard_negatives_are_not_flagged(text: str) -> None:
    assert types(text) == []


def test_secret_formats_are_detected() -> None:
    rng = random.Random(3)
    for _ in range(20):
        secret = make_secret(rng)
        assert types(f"config: {secret} end") == ["SECRET"], secret


def test_iban_wins_over_phone_and_card_inside_it() -> None:
    spans = DETECTOR.detect("Pay IT60 X054 2811 1010 0000 0123 456 now")
    assert [s.type for s in spans] == [PIIType.IBAN]
    assert spans[0].value == "IT60 X054 2811 1010 0000 0123 456"


def test_adjacent_values_are_all_found() -> None:
    text = "IBAN DE89370400440532013000 IBAN GB82 WEST 1234 5698 7654 32, cards 4111 1111 1111 1111 5500 0000 0000 0004"
    assert types(text) == ["IBAN", "IBAN", "CARD", "CARD"]


def test_existing_tokens_are_not_redetected() -> None:
    assert DETECTOR.detect("Contact <EMAIL_7f3a9c1e> or <PHONE_0123abcd>") == []


def test_tokens_have_typed_stable_format(vault: TokenVault) -> None:
    token = vault.tokenize("s1", PIIType.EMAIL, "giulia.rossi@example.com")
    assert re.fullmatch(r"<EMAIL_[0-9a-f]{8}>", token)
    assert TOKEN_RE.fullmatch(token)


def test_same_value_same_token_across_formats(vault: TokenVault) -> None:
    masker = Masker(vault)
    a = masker.mask("IBAN IT60 X054 2811 1010 0000 0123 456, mail Giulia.Rossi@Example.com, +39 347 123 4567", "s")
    b = masker.mask("IBAN IT60X0542811101000000123456, mail giulia.rossi@example.com, tel 347 123 4567", "s")
    assert [e.token for e in a.entities] == [e.token for e in b.entities]


def test_tokens_differ_across_sessions_and_secrets() -> None:
    v1, v2 = TokenVault(b"k1"), TokenVault(b"k2")
    t = v1.tokenize("s1", PIIType.EMAIL, "a@example.com")
    assert t != v1.tokenize("s2", PIIType.EMAIL, "a@example.com")
    assert t != v2.tokenize("s1", PIIType.EMAIL, "a@example.com")
    assert t == TokenVault(b"k1").tokenize("s1", PIIType.EMAIL, "a@example.com")  # deterministic given the key


_text = st.lists(
    st.sampled_from(["mail a.b@example.com", "IBAN DE89370400440532013000", "card 4111111111111111", "hello", "x",
                     "+39 347 123 4567", "born 12/03/1985", "<EMAIL_deadbeef>", " ", "\n", "192.168.0.1"]),
    max_size=8,
).map(" ".join)


@settings(max_examples=60, deadline=None)
@given(_text)
def test_masking_is_idempotent(text: str) -> None:
    masker = Masker(TokenVault(b"k"))
    once = masker.mask(text, "s").text
    assert masker.mask(once, "s").text == once


def test_rehydration_needs_scope_and_class(vault: TokenVault) -> None:
    masked = Masker(vault).mask("mail a@example.com, IBAN DE89370400440532013000", "s").text
    with pytest.raises(PermissionDenied):
        vault.rehydrate("s", masked, Principal(agent_id="nobody", data_classes=frozenset({PIIType.EMAIL})))
    email_only = Principal(agent_id="x", scopes=frozenset({"pii:reveal"}), data_classes=frozenset({PIIType.EMAIL}))
    revealed = vault.rehydrate("s", masked, email_only)
    assert "a@example.com" in revealed and "<IBAN_" in revealed


def test_forged_or_foreign_tokens_stay_untouched(vault: TokenVault) -> None:
    token = vault.tokenize("s1", PIIType.EMAIL, "a@example.com")
    boss = Principal(agent_id="x", scopes=frozenset({"pii:reveal"}), data_classes=frozenset(PIIType))
    assert vault.rehydrate("s2", token, boss) == token  # issued in another session
    assert vault.rehydrate("s1", "<EMAIL_00000000>", boss) == "<EMAIL_00000000>"


def test_truncated_hmac_collisions_get_longer_tokens() -> None:
    vault = TokenVault(b"k", token_hex=4)
    tokens = {vault.tokenize("s", PIIType.EMAIL, f"user{i}@example.com") for i in range(3000)}
    assert len(tokens) == 3000  # birthday collisions at 16 bits were resolved by lengthening
    assert any(len(t) > len("<EMAIL_0000>") for t in tokens)


def test_session_drop_and_lru_bound() -> None:
    vault = TokenVault(b"k", max_sessions=2)
    for s in ("a", "b", "c"):
        vault.tokenize(s, PIIType.EMAIL, "x@example.com")
    assert vault.size("a") == 0 and vault.size("c") == 1
    vault.drop_session("c")
    assert vault.size("c") == 0


def test_canonical_forms() -> None:
    assert canonical(PIIType.PHONE, "347 123 4567") == canonical(PIIType.PHONE, "+39 347 123 4567") == "+393471234567"
    assert canonical(PIIType.DATE_OF_BIRTH, "12 marzo 1985") == canonical(PIIType.DATE_OF_BIRTH, "1985-03-12")
    assert canonical(PIIType.IP_ADDRESS, "2001:DB8:0:0::1") == "2001:db8::1"


@pytest.mark.models("spacy")
def test_spacy_person_names_do_not_swallow_line_breaks() -> None:
    detector = PIIDetector(extra_finders=[SpacyNameFinder()])
    text = "Name: Giulia Rossi\nEmail: giulia@example.com"
    masked = Masker(TokenVault(b"k"), detector).mask(text, "s").text
    assert "Giulia" not in masked and "\nEmail: <EMAIL_" in masked
