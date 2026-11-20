"""Decode base64, hex and percent-encoded segments so the rules can read what they hide."""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
from dataclasses import dataclass

_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{20,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){10,}(?![0-9A-Fa-f])")
_HEX_ESCAPED = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){8,}")
_PERCENT = re.compile(r"(?:%[0-9A-Fa-f]{2}){8,}")


@dataclass(frozen=True)
class DecodedPayload:
    """Readable text recovered from an encoded segment."""

    encoding: str
    start: int
    end: int
    text: str


def _readable(data: bytes) -> str | None:
    """UTF-8 text that looks like language: mostly letters and spaces, at least two words."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) < 8:
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    letters = sum(ch.isalpha() or ch.isspace() for ch in text) / len(text)
    if printable < 0.95 or letters < 0.7 or len(text.split()) < 2:
        return None
    return text


def _b64(segment: str) -> bytes | None:
    padded = segment + "=" * (-len(segment) % 4)
    try:
        if "-" in segment or "_" in segment:
            return base64.urlsafe_b64decode(padded)
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def decode_payloads(text: str, *, depth: int = 2) -> list[DecodedPayload]:
    """Readable payloads hidden in ``text``, following nested encodings up to ``depth`` levels."""
    found: list[DecodedPayload] = []
    for encoding, pattern in (("base64", _BASE64), ("hex", _HEX), ("hex-escaped", _HEX_ESCAPED), ("percent", _PERCENT)):
        for m in pattern.finditer(text):
            segment = m.group()
            if encoding == "base64":
                data = _b64(segment)
            elif encoding == "hex":
                data = bytes.fromhex(segment)
            elif encoding == "hex-escaped":
                data = bytes.fromhex(segment.replace("\\x", ""))
            else:
                data = urllib.parse.unquote_to_bytes(segment)
            decoded = _readable(data) if data else None
            if decoded is None:
                continue
            found.append(DecodedPayload(encoding, m.start(), m.end(), decoded))
            if depth > 1:
                found.extend(
                    DecodedPayload(f"{encoding}>{inner.encoding}", m.start(), m.end(), inner.text)
                    for inner in decode_payloads(decoded, depth=depth - 1)
                )
    return found
