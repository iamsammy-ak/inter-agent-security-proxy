"""Checksum and structure validators that confirm regex candidates before anything is masked.

A regex alone would flag every 16-digit order number as a card and every ``IT``-prefixed code as an IBAN; these
validators keep precision high by checking the arithmetic each format carries.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from datetime import date

# ----------------------------------------------------------------------------------------------- payment cards


def luhn_valid(number: str) -> bool:
    """Luhn (mod 10) check used by payment card numbers."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def luhn_check_digit(partial: str) -> str:
    """Digit that makes ``partial + digit`` pass the Luhn check."""
    for candidate in "0123456789":
        if luhn_valid(partial + candidate):
            return candidate
    raise ValueError("unreachable: one digit always satisfies Luhn")


def card_brand(digits: str) -> str | None:
    """Card scheme from the issuer prefix and length, or ``None`` when no known scheme matches."""
    n = len(digits)
    if n < 13 or n > 19 or not digits.isdigit():
        return None
    p2, p3, p4, p6 = int(digits[:2]), int(digits[:3]), int(digits[:4]), int(digits[:6])
    if digits[0] == "4" and n in (13, 16, 19):
        return "visa"
    if n == 16 and (51 <= p2 <= 55 or 2221 <= p4 <= 2720):
        return "mastercard"
    if n == 15 and p2 in (34, 37):
        return "amex"
    if n in (16, 19) and (p4 == 6011 or p2 == 65 or 644 <= p3 <= 649 or 622126 <= p6 <= 622925):
        return "discover"
    if n in (16, 19) and 3528 <= p4 <= 3589:
        return "jcb"
    if n in (14, 16, 19) and (300 <= p3 <= 305 or p2 in (36, 38, 39)):
        return "diners"
    if 16 <= n <= 19 and p2 == 62:
        return "unionpay"
    if 12 <= n <= 19 and p4 in (5018, 5020, 5038, 5893, 6304, 6759, 6761, 6762, 6763):
        return "maestro"
    return None


def card_valid(value: str) -> bool:
    """A known card scheme prefix and length plus a passing Luhn check."""
    digits = re.sub(r"[ -]", "", value)
    return card_brand(digits) is not None and luhn_valid(digits)


# ------------------------------------------------------------------------------------------------------- IBAN

IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AT": 20, "BE": 16, "BG": 22, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "EE": 20,
    "ES": 24, "FI": 18, "FR": 27, "GB": 22, "GR": 27, "HR": 21, "HU": 28, "IE": 22, "IS": 26, "IT": 27,
    "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MT": 31, "NL": 18, "NO": 15, "PL": 28, "PT": 25,
    "RO": 24, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "VA": 22,
}  # fmt: skip


def compact_iban(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def iban_mod97(iban: str) -> int:
    """ISO 13616 remainder: move the first four characters to the end, map A-Z to 10-35, take mod 97."""
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97


def iban_check_digits(country: str, bban: str) -> str:
    """Check digits for a country code and basic bank account number."""
    return f"{98 - iban_mod97(country + '00' + bban):02d}"


def iban_valid(value: str) -> bool:
    """Country length table plus the mod-97 check (remainder must be 1)."""
    iban = compact_iban(value)
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban):
        return False
    expected = IBAN_LENGTHS.get(iban[:2])
    return expected == len(iban) and iban_mod97(iban) == 1


# ----------------------------------------------------------------------------------- Italian codice fiscale

CF_MONTHS = "ABCDEHLMPRST"
CF_OMOCODIA = "LMNPQRSTUV"  # letters that replace digits 0-9 when two people share a code
CF_DIGIT_POSITIONS = (6, 7, 9, 10, 12, 13, 14)
_CF_ODD = dict(
    zip(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 1, 0, 5, 7, 9, 13, 15, 17, 19, 21,
         2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23],
        strict=True,
    )
)  # fmt: skip
CF_PATTERN = re.compile(r"[A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]")


def _cf_even(ch: str) -> int:
    return int(ch) if ch.isdigit() else ord(ch) - ord("A")


def cf_check_char(first15: str) -> str:
    """Control character of a codice fiscale from its first 15 characters."""
    total = sum(_CF_ODD[ch] if i % 2 == 0 else _cf_even(ch) for i, ch in enumerate(first15.upper()))
    return chr(ord("A") + total % 26)


def cf_decode_omocodia(code: str) -> str:
    """Replace omocodia letters in the numeric positions with the digits they stand for."""
    chars = list(code.upper())
    for pos in CF_DIGIT_POSITIONS:
        if chars[pos] in CF_OMOCODIA:
            chars[pos] = str(CF_OMOCODIA.index(chars[pos]))
    return "".join(chars)


def cf_valid(value: str) -> bool:
    """Structure, birth month letter, birth day (1-31, or 41-71 for women) and the control character."""
    code = value.upper()
    if not CF_PATTERN.fullmatch(code):
        return False
    decoded = cf_decode_omocodia(code)
    day = int(decoded[9:11])
    if not (1 <= day <= 31 or 41 <= day <= 71):
        return False
    return cf_check_char(code[:15]) == code[15]


# ----------------------------------------------------------------------------------------------- network

def ipv4_valid(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def ipv6_valid(value: str) -> bool:
    if value.count(":") < 2:
        return False
    try:
        ipaddress.IPv6Address(value.split("%", 1)[0])
    except ValueError:
        return False
    return True


# ----------------------------------------------------------------------------------------------- dates


def plausible_birth_date(year: int, month: int, day: int, today: date | None = None) -> bool:
    """A real calendar date between 1900 and today."""
    today = today or date.today()
    try:
        born = date(year, month, day)
    except ValueError:
        return False
    return date(1900, 1, 1) <= born <= today


# ----------------------------------------------------------------------------------------------- secrets


def shannon_entropy(value: str) -> float:
    """Bits per character; random API keys score above 3.5, English words around 2.5-3."""
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in counts.values())
