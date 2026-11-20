"""Seeded synthetic data for the benchmarks. Every value here is invented.

* :func:`pii_corpus` builds English and Italian sentences with known PII spans (valid checksums, Italian formats)
  and hard negatives that look like PII but are not (Luhn-failing numbers, wrong IBAN check digits, dates without
  a birth context, version strings, UUIDs, hashes).
* :func:`agentic_suite` crosses hand-written attack instructions with obfuscations seen in real indirect injections
  (base64, hex, tag characters, zero-width splitting, homoglyphs, HTML comments, bidi controls) and pairs them with
  hand-written benign agent traffic. The rules' author also wrote these seeds, so results on this suite are
  optimistic for the rules; it measures coverage of known tricks, not generalisation.
"""

from __future__ import annotations

import base64
import json
import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources

from .models import PIIType
from .pii.validators import CF_MONTHS, cf_check_char, iban_check_digits, luhn_check_digit

# ----------------------------------------------------------------------------------------------- PII values

FIRST = ["Giulia", "Marco", "Sara", "Luca", "Francesca", "Alessandro", "Chiara", "Matteo", "Emma", "Oliver",
         "Sophie", "James", "Amelia", "Noah", "Elena", "Davide"]
LAST = ["Rossi", "Bianchi", "Esposito", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Smith", "Taylor",
        "Brown", "Wilson", "Conti", "Gallo", "Fontana", "Moretti"]
DOMAINS = ["example.com", "example.org", "example.net", "mail.example.it", "corp.example.co.uk"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
MONTHS_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
             "ottobre", "novembre", "dicembre"]
BELFIORE = ["H501", "F205", "L219", "A944", "G273", "F839", "D612", "L736", "C351", "B157"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _group(value: str, size: int = 4, sep: str = " ") -> str:
    return sep.join(value[i : i + size] for i in range(0, len(value), size))


def make_email(rng: random.Random) -> str:
    sep = rng.choice([".", "_", ""])
    return f"{rng.choice(FIRST).lower()}{sep}{rng.choice(LAST).lower()}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}"


def make_phone(rng: random.Random) -> str:
    kind = rng.choice(["it-mobile", "it-mobile", "it-landline", "intl", "nanp"])
    if kind == "it-mobile":
        n = "3" + rng.choice("123456789") + _digits(rng, 8)
        return rng.choice([f"+39 {n[:3]} {n[3:6]} {n[6:]}", f"{n[:3]} {n[3:]}", f"+39{n}", f"0039 {n[:3]} {n[3:]}",
                           f"{n[:3]}-{n[3:6]}-{n[6:]}"])
    if kind == "it-landline":
        prefix = rng.choice(["02", "06", "011", "055", "081", "051"])
        rest = _digits(rng, 10 - len(prefix))
        return rng.choice([f"{prefix} {rest}", f"+39 {prefix} {rest[:3]} {rest[3:]}", f"{prefix}/{rest}"])
    if kind == "intl":
        cc, body = rng.choice([("44", "20 7946 " + _digits(rng, 4)), ("49", "30 " + _digits(rng, 7)),
                               ("33", "1 " + " ".join(_digits(rng, 2) for _ in range(4))),
                               ("34", "91 " + _digits(rng, 3) + " " + _digits(rng, 4))])
        return f"+{cc} {body}"
    area = str(rng.randint(2, 9)) + _digits(rng, 2)
    return rng.choice([f"({area}) 555-{_digits(rng, 4)}", f"+1 {area} 555 {_digits(rng, 4)}"])


_IBAN_BBAN: dict[str, Callable[[random.Random], str]] = {
    "IT": lambda r: r.choice(string.ascii_uppercase) + _digits(r, 10) + _digits(r, 12),
    "DE": lambda r: _digits(r, 18),
    "FR": lambda r: _digits(r, 23),
    "ES": lambda r: _digits(r, 20),
    "GB": lambda r: "".join(r.choice(string.ascii_uppercase) for _ in range(4)) + _digits(r, 14),
    "NL": lambda r: "".join(r.choice(string.ascii_uppercase) for _ in range(4)) + _digits(r, 10),
}


def make_iban(rng: random.Random, *, valid: bool = True) -> str:
    country = rng.choice(["IT", "IT", "DE", "FR", "ES", "GB", "NL"])
    bban = _IBAN_BBAN[country](rng)
    check = iban_check_digits(country, bban)
    if not valid:
        check = f"{(int(check) + rng.randint(1, 90)) % 97 + 2:02d}"
        if check == iban_check_digits(country, bban):
            check = f"{(int(check) + 1) % 97 + 2:02d}"
    iban = f"{country}{check}{bban}"
    return _group(iban) if rng.random() < 0.6 else iban


def make_card(rng: random.Random, *, valid: bool = True) -> str:
    prefix, length = rng.choice([("4", 16), ("51", 16), ("55", 16), ("2221", 16), ("34", 15), ("37", 15), ("6011", 16)])
    partial = prefix + _digits(rng, length - len(prefix) - 1)
    check = luhn_check_digit(partial)
    if not valid:
        check = str((int(check) + rng.randint(1, 9)) % 10)
    number = partial + check
    if length == 15:
        return rng.choice([number, f"{number[:4]} {number[4:10]} {number[10:]}"])
    return rng.choice([number, _group(number), _group(number, sep="-")])


def make_codice_fiscale(rng: random.Random, *, valid: bool = True) -> str:
    letters = "".join(rng.choice("BCDFGLMNPRSTVZ") for _ in range(3))
    letters += "".join(rng.choice("BCDFGLMNPRSTVZA") for _ in range(3))
    year = f"{rng.randint(40, 99):02d}"
    month = rng.choice(CF_MONTHS)
    day = rng.randint(1, 28) + (40 if rng.random() < 0.5 else 0)
    first15 = f"{letters}{year}{month}{day:02d}{rng.choice(BELFIORE)}"
    if rng.random() < 0.1:  # omocodia: a digit replaced by its letter
        pos = rng.choice([6, 7, 12, 13, 14])
        first15 = first15[:pos] + "LMNPQRSTUV"[int(first15[pos])] + first15[pos + 1 :]
    check = cf_check_char(first15)
    if not valid:
        check = chr((ord(check) - 65 + rng.randint(1, 25)) % 26 + 65)
    return first15 + check


def make_ip(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return ".".join(str(rng.randint(1, 254)) for _ in range(4))
    groups = ["2001", "db8"] + [f"{rng.randint(0, 0xFFFF):x}" for _ in range(6)]
    full = ":".join(groups)
    return full if rng.random() < 0.5 else f"2001:db8::{groups[-2]}:{groups[-1]}"


def make_dob(rng: random.Random, lang: str) -> str:
    y, m, d = rng.randint(1940, 2005), rng.randint(1, 12), rng.randint(1, 28)
    formats = [f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{d:02d}.{m:02d}.{y}"]
    formats.append(f"{d} {MONTHS_IT[m - 1]} {y}" if lang == "it" else rng.choice([f"{d} {MONTHS_EN[m - 1]} {y}",
                                                                                  f"{MONTHS_EN[m - 1]} {d}, {y}"]))
    return rng.choice(formats)


def make_secret(rng: random.Random) -> str:
    alnum = string.ascii_letters + string.digits

    def rand(n: int) -> str:
        return "".join(rng.choice(alnum) for _ in range(n))

    return rng.choice([
        "sk-" + "proj-" + rand(40),
        "gh" + "p_" + rand(36),
        "AK" + "IA" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16)),
        "xo" + "xb-" + _digits(rng, 11) + "-" + rand(24),
        "ey" + "J" + rand(20) + ".ey" + "J" + rand(30) + "." + rand(32),
        "sk" + "_live_" + rand(24),
    ])


# ----------------------------------------------------------------------------------------------- PII corpus

TEMPLATES_EN = [
    "Please contact {PERSON} at {EMAIL} or on {PHONE} about order {ORDER}.",
    "Refund approved: send 49.90 EUR to IBAN {IBAN} by {PLAIN_DATE}.",
    "Card on file {CARD}, expiry 08/27; previous card {BAD_CARD} was declined.",
    "Customer {PERSON}, born on {DOB}, phone {PHONE}.",
    "Login from {IP} at 10:42:07 (build {VERSION}, request {UUID}).",
    "Deployment config leaked: {SECRET}",
    "Ticket {ORDER}: user {EMAIL} reports duplicate charge on {CARD}.",
    "DOB: {DOB}. Contact: {EMAIL}. Codice fiscale {CF}.",
    "Wire transfer reference {BAD_IBAN} failed validation; resend to {IBAN}.",
    "Commit {SHA} fixed the timeout seen from {IP}.",
    "Call {PHONE} after 2pm; invoice total {PRICE}.",
    "Meeting moved to {PLAIN_DATE} at 15:30 in room 4.12.",
]
TEMPLATES_IT = [
    "Il cliente {PERSON}, nato il {DOB}, codice fiscale {CF}, tel. {PHONE}.",
    "Bonifico su IBAN {IBAN} entro il {PLAIN_DATE}, causale fattura {ORDER}.",
    "Scrivere a {EMAIL} per la fattura n. {ORDER} di importo {PRICE}.",
    "Codice fiscale dichiarato {BAD_CF} non valido; quello corretto è {CF}.",
    "Data di nascita: {DOB}. Cellulare {PHONE}.",
    "Accesso dall'indirizzo {IP} alle 09:15, versione app {VERSION}.",
    "La carta {CARD} è stata addebitata; il numero ordine {BAD_CARD} non è una carta.",
    "Chiave API nel log: {SECRET}",
    "Recapito della signora {PERSON}: {EMAIL}, {PHONE}.",
    "Riunione fissata per il {PLAIN_DATE}, sala 3.",
]

_FILLERS: dict[str, tuple[PIIType | None, Callable[[random.Random, str], str]]] = {
    "EMAIL": (PIIType.EMAIL, lambda r, lang: make_email(r)),
    "PHONE": (PIIType.PHONE, lambda r, lang: make_phone(r)),
    "IBAN": (PIIType.IBAN, lambda r, lang: make_iban(r)),
    "CARD": (PIIType.CARD, lambda r, lang: make_card(r)),
    "CF": (PIIType.CODICE_FISCALE, lambda r, lang: make_codice_fiscale(r)),
    "IP": (PIIType.IP_ADDRESS, lambda r, lang: make_ip(r)),
    "DOB": (PIIType.DATE_OF_BIRTH, make_dob),
    "SECRET": (PIIType.SECRET, lambda r, lang: make_secret(r)),
    "PERSON": (PIIType.PERSON, lambda r, lang: f"{r.choice(FIRST)} {r.choice(LAST)}"),
    # hard negatives: look like PII, are not
    "BAD_CARD": (None, lambda r, lang: make_card(r, valid=False)),
    "BAD_IBAN": (None, lambda r, lang: make_iban(r, valid=False)),
    "BAD_CF": (None, lambda r, lang: make_codice_fiscale(r, valid=False)),
    "ORDER": (None, lambda r, lang: r.choice(["INV-", "ORD-", "", "n. "]) + _digits(r, r.choice([6, 8, 10]))),
    "PLAIN_DATE": (None, lambda r, lang: r.choice([f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/2026",
                                                   f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"])),
    "VERSION": (None, lambda r, lang: f"{r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}"),
    "UUID": (None, lambda r, lang: "-".join("".join(r.choice("0123456789abcdef") for _ in range(n))
                                            for n in (8, 4, 4, 4, 12))),
    "SHA": (None, lambda r, lang: "".join(r.choice("0123456789abcdef") for _ in range(40))),
    "PRICE": (None, lambda r, lang: f"€{r.randint(1, 9)}.{r.randint(100, 999)},{r.randint(10, 99)}"),
}


VARIANT_RATE = 0.08


def _variant(name: str, rng: random.Random, value: str) -> str | None:
    """Real-world spellings the detectors do not target, so the benchmark also measures known gaps."""
    digits = "".join(c for c in value if c.isdigit())
    if name == "EMAIL":
        local, domain = value.split("@", 1)
        return f"{local} [at] {domain.replace('.', ' [dot] ')}"
    if name == "PHONE":
        n = "3" + rng.choice("123456789") + _digits(rng, 8)
        return f"{n[:3]}.{n[3:6]}.{n[6:8]}.{n[8:]}"
    if name == "IBAN":
        return value.replace(" ", "").lower()
    if name == "CARD" and len(digits) == 16:
        return ".".join(digits[i : i + 4] for i in range(0, 16, 4))
    if name == "DOB":
        y, m, d = rng.randint(1940, 2005), rng.randint(1, 12), rng.randint(1, 28)
        return f"{d:02d}-{MONTHS_EN[m - 1][:3]}-{y}"
    return None


@dataclass
class SynthText:
    """A synthetic sentence with its gold PII spans ``(type, start, end)``."""

    text: str
    lang: str
    spans: list[tuple[PIIType, int, int]] = field(default_factory=list)
    negatives: int = 0
    variants: int = 0


def pii_corpus(n: int = 600, seed: int = 7) -> list[SynthText]:
    """``n`` sentences (about 55% English, 45% Italian) with gold spans. About 8% of e-mail, phone, IBAN, card
    and birth-date values use a spelling the detectors do not target (see :func:`_variant`)."""
    rng = random.Random(seed)
    corpus: list[SynthText] = []
    for _ in range(n):
        lang = "it" if rng.random() < 0.45 else "en"
        template = rng.choice(TEMPLATES_IT if lang == "it" else TEMPLATES_EN)
        out, spans, negatives, variants, cursor = [], [], 0, 0, 0
        for part in string.Formatter().parse(template):
            literal, name = part[0], part[1]
            out.append(literal)
            cursor += len(literal)
            if name:
                pii_type, make = _FILLERS[name]
                value = make(rng, lang)
                if pii_type is not None and rng.random() < VARIANT_RATE:
                    variant = _variant(name, rng, value)
                    if variant is not None:
                        value, variants = variant, variants + 1
                if pii_type is not None:
                    spans.append((pii_type, cursor, cursor + len(value)))
                else:
                    negatives += 1
                out.append(value)
                cursor += len(value)
        corpus.append(SynthText("".join(out), lang, spans, negatives, variants))
    return corpus


# ----------------------------------------------------------------------------------------------- agentic suite


def _tags(text: str) -> str:
    return "".join(chr(0xE0000 + ord(c)) for c in text if 0x20 <= ord(c) <= 0x7E)


def _zero_width(text: str) -> str:
    zw = "​"
    return " ".join(zw.join(word) if len(word) > 4 else word for word in text.split(" "))


_HOMOGLYPHS = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "i": "і"})

TRANSFORMS: dict[str, Callable[[str, str], str]] = {
    "plain": lambda attack, cover: attack,
    "embedded": lambda attack, cover: f"{cover}\n\n{attack}\n\nEnd of record.",
    "html-comment": lambda attack, cover: f"{cover}\n<!-- {attack} -->",
    "base64": lambda attack, cover: f"{cover}\nAttachment (base64): {base64.b64encode(attack.encode()).decode()}",
    "hex": lambda attack, cover: f"{cover}\nChecksum payload: {attack.encode().hex()}",
    "tag-chars": lambda attack, cover: f"{cover}{_tags(attack)}",
    "zero-width": lambda attack, cover: f"{cover}\n{_zero_width(attack)}",
    "homoglyph": lambda attack, cover: f"{cover}\n{attack.translate(_HOMOGLYPHS)}",
    "fullwidth": lambda attack, cover: f"{cover}\n" + "".join(
        chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in attack),
    "bidi": lambda attack, cover: f"{cover}\n‮{attack[::-1]}‬",
}


@dataclass
class SuiteItem:
    """One synthetic agent-to-agent payload."""

    text: str
    label: int
    transform: str
    seed_id: int


def _seeds() -> dict[str, list[str]]:
    raw = (resources.files("agent_proxy") / "data" / "agentic_seeds.json").read_text(encoding="utf-8")
    seeds: dict[str, list[str]] = json.loads(raw)
    return seeds


def agentic_suite() -> list[SuiteItem]:
    """Attack seeds under every transform (label 1) plus benign agent traffic (label 0)."""
    seeds = _seeds()
    covers = seeds["covers"]
    items: list[SuiteItem] = []
    for i, attack in enumerate(seeds["attacks"]):
        cover = covers[i % len(covers)]
        for name, transform in TRANSFORMS.items():
            items.append(SuiteItem(transform(attack, cover), 1, name, i))
    items.extend(SuiteItem(text, 0, "benign", i) for i, text in enumerate(seeds["benign"]))
    return items
