from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_proxy.pii.validators import (
    IBAN_LENGTHS,
    card_brand,
    card_valid,
    cf_check_char,
    cf_valid,
    iban_check_digits,
    iban_valid,
    ipv4_valid,
    ipv6_valid,
    luhn_check_digit,
    luhn_valid,
    plausible_birth_date,
    shannon_entropy,
)


@pytest.mark.parametrize("number", ["4111111111111111", "5500 0000 0000 0004", "378282246310005", "6011000990139424"])
def test_known_test_cards_pass(number: str) -> None:
    assert card_valid(number)


def test_luhn_failure_and_unknown_prefix_are_rejected() -> None:
    assert not card_valid("4111111111111112")  # one digit off
    assert luhn_valid("1234567812345670") and not card_valid("1234567812345670")  # Luhn ok, no scheme
    assert card_brand("4111111111111111") == "visa"
    assert card_brand("378282246310005") == "amex"


@settings(deadline=None)
@given(st.text(alphabet=string.digits, min_size=11, max_size=18))
def test_luhn_check_digit_always_validates(partial: str) -> None:
    assert luhn_valid(partial + luhn_check_digit(partial))


@pytest.mark.parametrize(
    "iban", ["IT60 X054 2811 1010 0000 0123 456", "DE89370400440532013000", "GB82 WEST 1234 5698 7654 32"]
)
def test_published_example_ibans_pass(iban: str) -> None:
    assert iban_valid(iban)


def test_iban_wrong_check_digits_or_length_fail() -> None:
    assert not iban_valid("IT61X0542811101000000123456")
    assert not iban_valid("IT60X054281110100000012345")  # one character short
    assert not iban_valid("XX60X0542811101000000123456")  # unknown country


@settings(deadline=None)
@given(st.sampled_from(sorted(IBAN_LENGTHS)), st.randoms(use_true_random=False))
def test_computed_check_digits_make_valid_ibans(country: str, rng) -> None:  # type: ignore[no-untyped-def]
    bban = "".join(rng.choice(string.digits) for _ in range(IBAN_LENGTHS[country] - 4))
    assert iban_valid(country + iban_check_digits(country, bban) + bban)


def test_codice_fiscale_checksum_month_and_day() -> None:
    first15 = "RSSMRA85T10A562"
    code = first15 + cf_check_char(first15)
    assert code == "RSSMRA85T10A562S"
    assert cf_valid(code) and cf_valid(code.lower())
    assert not cf_valid(first15 + "A")  # wrong control character
    bad_day = "RSSMRA85T35A562"
    assert not cf_valid(bad_day + cf_check_char(bad_day))  # day 35 does not exist (women use 41-71)
    woman = "RSSGLI85C52H501"
    assert cf_valid(woman + cf_check_char(woman))


def test_codice_fiscale_omocodia_is_accepted() -> None:
    # Two people with the same code: the last digit of the town code is replaced by its letter (1 -> M).
    first15 = "RSSMRA85T10A56M"
    assert cf_valid(first15 + cf_check_char(first15))


def test_ip_and_date_validators() -> None:
    assert ipv4_valid("192.168.1.20") and not ipv4_valid("256.1.1.1")
    assert ipv6_valid("2001:db8::8a2e:370:7334") and not ipv6_valid("12:30:45")
    assert plausible_birth_date(1985, 3, 12) and not plausible_birth_date(1985, 2, 30)
    assert not plausible_birth_date(1850, 1, 1)


def test_entropy_separates_keys_from_words() -> None:
    assert shannon_entropy("Zx9kLm2Qp7Rt4Vw8") > 3.5
    assert shannon_entropy("aaaaaaaaaaaa") == 0.0
