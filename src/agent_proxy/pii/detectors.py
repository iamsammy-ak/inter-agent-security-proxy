"""Regex candidates confirmed by validators, one detector per PII class.

Each detector finds candidates with a deliberately loose regular expression and keeps only those its validator
accepts (Luhn, mod-97, the codice fiscale control character, ``ipaddress`` parsing, calendar checks or nearby
context words). Overlaps between classes are resolved by priority, so an IBAN is never also reported as a phone
number.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date

from ..models import PIIType
from .validators import (
    IBAN_LENGTHS,
    card_valid,
    cf_valid,
    iban_valid,
    ipv4_valid,
    ipv6_valid,
    plausible_birth_date,
    shannon_entropy,
)

TOKEN_RE = re.compile(r"<([A-Z][A-Z_]*)_([0-9a-f]{4,64})>")

PRIORITY: dict[PIIType, int] = {
    PIIType.SECRET: 90,
    PIIType.IBAN: 80,
    PIIType.CARD: 70,
    PIIType.CODICE_FISCALE: 65,
    PIIType.EMAIL: 60,
    PIIType.IP_ADDRESS: 50,
    PIIType.PHONE: 40,
    PIIType.DATE_OF_BIRTH: 30,
    PIIType.PERSON: 10,
}


@dataclass(frozen=True)
class PIISpan:
    """A confirmed PII value at ``text[start:end]``; ``detector`` names the check that confirmed it."""

    type: PIIType
    start: int
    end: int
    value: str
    detector: str


Finder = Callable[[str], Iterator[PIISpan]]

# ----------------------------------------------------------------------------------------------- email

_EMAIL = re.compile(
    r"(?<![\w.+%-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}(?![A-Za-z0-9-])"
)


def find_emails(text: str) -> Iterator[PIISpan]:
    for m in _EMAIL.finditer(text):
        local = m.group().split("@", 1)[0]
        if ".." in local or local.endswith(".") or len(m.group()) > 254:
            continue
        yield PIISpan(PIIType.EMAIL, m.start(), m.end(), m.group(), "email:structure")


# ----------------------------------------------------------------------------------------------- phone

_PHONE_CONTEXT = re.compile(
    r"\b(tel|tel\.|phone|telephone|telefono|cell|cellulare|mobile|whatsapp|fax|call|chiama\w*|numero|number|"
    r"contact|contatto|recapito|sms)\b",
    re.I,
)
_PHONE_INTL = re.compile(r"(?<![\w+])(?:\+|00)[1-9][\d \-.()/]{6,20}\d(?!\w)")
_PHONE_IT_MOBILE = re.compile(r"(?<![\w+.,/-])3\d{2}[ .-]?\d{3}[ .-]?\d{3,4}(?![\w.,/-]?\d)")
_PHONE_IT_LANDLINE = re.compile(r"(?<![\w+.,/-])0\d{1,3}[ .\-/]?\d{3,4}[ .-]?\d{2,4}(?![\w.,/-]?\d)")
_PHONE_NANP = re.compile(r"(?<![\w+])(?:\([2-9]\d{2}\)\s?|[2-9]\d{2}[-.\s])\d{3}[-.\s]\d{4}(?!\d)")


def _has_context(text: str, start: int, pattern: re.Pattern[str], window: int = 40) -> bool:
    return bool(pattern.search(text[max(0, start - window) : start]))


def find_phones(text: str) -> Iterator[PIISpan]:
    for m in _PHONE_INTL.finditer(text):
        raw = m.group()
        digits = re.sub(r"\D", "", raw)
        if raw.startswith("00"):
            digits = digits[2:]
            if raw.isdigit() and not _has_context(text, m.start(), _PHONE_CONTEXT):
                continue  # "0012345678" is more often a reference number than a dialled call
        if 8 <= len(digits) <= 15 and not re.search(r"[ .\-/]{2}", raw):
            yield PIISpan(PIIType.PHONE, m.start(), m.end(), raw, "phone:e164")
    for pattern, name, lengths in (
        (_PHONE_IT_MOBILE, "phone:it-mobile", (9, 10)),
        (_PHONE_IT_LANDLINE, "phone:it-landline", (8, 9, 10, 11)),
    ):
        for m in pattern.finditer(text):
            raw = m.group()
            digits = re.sub(r"\D", "", raw)
            formatted = raw != digits
            if len(digits) in lengths and (formatted or _has_context(text, m.start(), _PHONE_CONTEXT)):
                yield PIISpan(PIIType.PHONE, m.start(), m.end(), raw, name)
    for m in _PHONE_NANP.finditer(text):
        yield PIISpan(PIIType.PHONE, m.start(), m.end(), m.group(), "phone:nanp")


# ----------------------------------------------------------------------------------------------- IBAN

_IBAN_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,32}")


def _iban_shape_end(text: str, m: re.Match[str]) -> int | None:
    """End of an IBAN-shaped string (known country code, right length) starting at the candidate, valid or not.
    The candidate may run into the next word ("... 0123 456 BIC"), so it is cut at the country's length."""
    expected = IBAN_LENGTHS.get(m.group()[:2])
    if expected is None:
        return None
    count = 0
    for i in range(m.start(), m.end()):
        if text[i] != " ":
            count += 1
        if count == expected:
            end = i + 1
            return None if end < len(text) and text[end].isalnum() else end
    return None


def _iban_at(text: str, m: re.Match[str]) -> int | None:
    """End of a valid IBAN starting at the candidate, or ``None``."""
    end = _iban_shape_end(text, m)
    return end if end is not None and iban_valid(text[m.start() : end]) else None


def iban_shapes(text: str) -> list[tuple[int, int]]:
    """Spans that look like IBANs even when the checksum fails; digits inside them are not cards or phones."""
    shapes = []
    for m in _IBAN_CANDIDATE.finditer(text):
        end = _iban_shape_end(text, m)
        if end is not None:
            shapes.append((m.start(), end))
    return shapes


def find_ibans(text: str) -> Iterator[PIISpan]:
    pos = 0
    while (m := _IBAN_CANDIDATE.search(text, pos)) is not None:
        end = _iban_at(text, m)
        if end is None:
            pos = m.start() + 1
            continue
        yield PIISpan(PIIType.IBAN, m.start(), end, text[m.start() : end], "iban:mod97")
        pos = end  # resume right after it, so a second IBAN swallowed by the candidate is still found


# ----------------------------------------------------------------------------------------------- cards

_CARD_CANDIDATE = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,22}(?![\d])")
_CARD_LENGTHS = (16, 15, 14, 19, 13, 17, 18)


def _card_at(text: str, m: re.Match[str]) -> int | None:
    positions = [i for i in range(m.start(), m.end()) if text[i].isdigit()]
    for length in _CARD_LENGTHS:
        if length > len(positions):
            continue
        end = positions[length - 1] + 1
        if length < len(positions) and text[end] not in " -":
            continue  # a shorter prefix must end at a separator, not mid-number
        if card_valid(text[m.start() : end]):
            return end
    return None


def find_cards(text: str) -> Iterator[PIISpan]:
    pos = 0
    while (m := _CARD_CANDIDATE.search(text, pos)) is not None:
        end = _card_at(text, m)
        if end is None:
            pos = m.start() + 1
            continue
        yield PIISpan(PIIType.CARD, m.start(), end, text[m.start() : end], "card:luhn")
        pos = end


# ----------------------------------------------------------------------------------- codice fiscale

_CF_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{6}[0-9LMNPQRSTUVlmnpqrstuv]{2}[A-Za-z][0-9LMNPQRSTUVlmnpqrstuv]{2}[A-Za-z]"
    r"[0-9LMNPQRSTUVlmnpqrstuv]{3}[A-Za-z](?![A-Za-z0-9])"
)


def find_codici_fiscali(text: str) -> Iterator[PIISpan]:
    for m in _CF_CANDIDATE.finditer(text):
        if cf_valid(m.group()):
            yield PIISpan(PIIType.CODICE_FISCALE, m.start(), m.end(), m.group(), "cf:checksum")


# ----------------------------------------------------------------------------------------------- IP

_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)")
_IPV6 = re.compile(
    r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9A-Fa-f]{0,4})(?:%[\w.]+)?(?![\w:])"
)


def find_ips(text: str) -> Iterator[PIISpan]:
    for m in _IPV4.finditer(text):
        if ipv4_valid(m.group()):
            yield PIISpan(PIIType.IP_ADDRESS, m.start(), m.end(), m.group(), "ip:v4")
    for m in _IPV6.finditer(text):
        value = m.group()
        if sum(ch.isalnum() for ch in value) >= 3 and ipv6_valid(value):
            yield PIISpan(PIIType.IP_ADDRESS, m.start(), m.end(), value, "ip:v6")


# ----------------------------------------------------------------------------------------------- dates of birth

_MONTHS = {
    name: i + 1
    for names in (
        "january february march april may june july august september october november december",
        "jan feb mar apr may jun jul aug sep oct nov dec",
        "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre",
    )
    for i, name in enumerate(names.split())
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DOB_CONTEXT = re.compile(
    r"\b(born|birth|birthday|dob|d\.o\.b|nato|nata|nascita|geboren|geburtsdatum)\b", re.I
)
_DATE_NUMERIC = re.compile(r"(?<![\d/.-])(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?![\d/.-]?\d)")
_DATE_ISO = re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d-]?\d)")
_DATE_DMY_TEXT = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I)
_DATE_MDY_TEXT = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)


def find_dates_of_birth(text: str, today: date | None = None) -> Iterator[PIISpan]:
    """Dates count as PII only with a birth keyword in the preceding 40 characters ("born", "DOB", "nato il")."""
    for m in _DATE_NUMERIC.finditer(text):
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        day, month = (b, a) if b > 12 >= a else (a, b)  # day-first unless only month-first is possible
        if plausible_birth_date(year, month, day, today) and _has_context(text, m.start(), _DOB_CONTEXT):
            yield PIISpan(PIIType.DATE_OF_BIRTH, m.start(), m.end(), m.group(), "dob:numeric")
    for m in _DATE_ISO.finditer(text):
        year, month, day = (int(g) for g in m.groups())
        if plausible_birth_date(year, month, day, today) and _has_context(text, m.start(), _DOB_CONTEXT):
            yield PIISpan(PIIType.DATE_OF_BIRTH, m.start(), m.end(), m.group(), "dob:iso")
    for pattern, order in ((_DATE_DMY_TEXT, "dmy"), (_DATE_MDY_TEXT, "mdy")):
        for m in pattern.finditer(text):
            if order == "dmy":
                day, month, year = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
            else:
                month, day, year = _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
            if plausible_birth_date(year, month, day, today) and _has_context(text, m.start(), _DOB_CONTEXT):
                yield PIISpan(PIIType.DATE_OF_BIRTH, m.start(), m.end(), m.group(), "dob:text")


# ----------------------------------------------------------------------------------------------- secrets

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("secret:pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{16,}?-----END [A-Z ]*PRIVATE KEY-----")),
    ("secret:jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("secret:sk", re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{20,}")),
    ("secret:stripe-style", re.compile(r"\b[spr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("secret:github", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("secret:aws-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("secret:slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("secret:hf", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("secret:google", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|secret(?:[_-]?key)?|access[_-]?key|client[_-]?secret|auth[_-]?token|token|"
    r"password|passwd|pwd)\b[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-./+=]{12,})"
)


def find_secrets(text: str) -> Iterator[PIISpan]:
    for name, pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            yield PIISpan(PIIType.SECRET, m.start(), m.end(), m.group(), name)
    for m in _SECRET_ASSIGNMENT.finditer(text):
        value = m.group(1)
        if TOKEN_RE.fullmatch(f"<{value}>") or value.startswith("<"):
            continue
        has_mix = any(c.isdigit() for c in value) and any(c.isalpha() for c in value)
        if has_mix and shannon_entropy(value) >= 3.0:
            yield PIISpan(PIIType.SECRET, m.start(1), m.end(1), value, "secret:assignment")


# ----------------------------------------------------------------------------------------------- registry

DEFAULT_FINDERS: dict[PIIType, Finder] = {
    PIIType.SECRET: find_secrets,
    PIIType.IBAN: find_ibans,
    PIIType.CARD: find_cards,
    PIIType.CODICE_FISCALE: find_codici_fiscali,
    PIIType.EMAIL: find_emails,
    PIIType.IP_ADDRESS: find_ips,
    PIIType.PHONE: find_phones,
    PIIType.DATE_OF_BIRTH: find_dates_of_birth,
}


_NUMERIC_TYPES = frozenset({PIIType.CARD, PIIType.PHONE})


def resolve_overlaps(spans: Iterable[PIISpan]) -> list[PIISpan]:
    """Keep the highest-priority (then longest) span wherever candidates overlap; return them in text order."""
    ranked = sorted(spans, key=lambda s: (-PRIORITY[s.type], -(s.end - s.start), s.start))
    chosen: list[PIISpan] = []
    for span in ranked:
        if all(span.end <= c.start or span.start >= c.end for c in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: s.start)


class PIIDetector:
    """Runs the enabled finders, drops anything inside an existing token, and resolves overlaps."""

    def __init__(
        self,
        types: Iterable[PIIType] | None = None,
        extra_finders: Iterable[Finder] = (),
    ) -> None:
        enabled = set(types) if types is not None else set(DEFAULT_FINDERS)
        self.finders: list[Finder] = [f for t, f in DEFAULT_FINDERS.items() if t in enabled]
        self.finders.extend(extra_finders)

    def detect(self, text: str) -> list[PIISpan]:
        tokens = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        shapes = iban_shapes(text)
        candidates = [
            span
            for finder in self.finders
            for span in finder(text)
            if all(span.end <= s or span.start >= e for s, e in tokens)
            and not (span.type in _NUMERIC_TYPES and any(s <= span.start and span.end <= e for s, e in shapes))
        ]
        return resolve_overlaps(candidates)


def normalise_date(value: str) -> str:
    """ISO form of a date string matched by the birth-date patterns (the input unchanged if it does not parse)."""
    for pattern, kind in ((_DATE_ISO, "iso"), (_DATE_NUMERIC, "num"), (_DATE_DMY_TEXT, "dmy"), (_DATE_MDY_TEXT, "mdy")):
        m = pattern.fullmatch(value.strip())
        if not m:
            continue
        g = m.groups()
        if kind == "iso":
            year, month, day = int(g[0]), int(g[1]), int(g[2])
        elif kind == "num":
            a, b, year = int(g[0]), int(g[1]), int(g[2])
            day, month = (b, a) if b > 12 >= a else (a, b)
        elif kind == "dmy":
            day, month, year = int(g[0]), _MONTHS[g[1].lower()], int(g[2])
        else:
            month, day, year = _MONTHS[g[0].lower()], int(g[1]), int(g[2])
        return f"{year:04d}-{month:02d}-{day:02d}"
    return value.strip()
