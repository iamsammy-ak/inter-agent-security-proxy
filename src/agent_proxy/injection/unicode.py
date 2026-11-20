"""Find and strip characters a human reviewer cannot see but a model can read.

* **Tag characters** (U+E0000-U+E007F) mirror ASCII and render as nothing ("ASCII smuggling"). The one legitimate
  use, subdivision flag emoji such as England's, is recognised and not reported.
* **Variation-selector runs**: one selector after an emoji is normal; a run of them can carry arbitrary bytes.
* **Zero-width characters** split words so keyword filters miss them. Joiners next to non-ASCII text (emoji
  sequences, Indic and Persian scripts) are kept.
* **Bidirectional controls** reorder how text displays ("Trojan Source").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

TAG_FIRST, TAG_LAST, TAG_CANCEL = 0xE0000, 0xE007F, 0xE007F
BLACK_FLAG = 0x1F3F4
ZERO_WIDTH = frozenset({0x00AD, 0x180E, 0x200B, 0x200C, 0x200D, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF})
JOINERS = frozenset({0x200C, 0x200D})
BIDI_CONTROLS = frozenset({*range(0x202A, 0x202F), *range(0x2066, 0x206A)})
_FLAG_TAGS = re.compile(r"[a-z0-9]{2,7}")
DATAMARK = "\u02c6"  # spotlighting word separator; read as a space when matching rules

# Cyrillic and Greek letters that render like Latin ones, mapped to the Latin letter they imitate.
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0443": "y", "\u0445": "x",
        "\u0456": "i", "\u0458": "j", "\u0455": "s", "\u0501": "d", "\u04bb": "h", "\u04cf": "l", "\u051b": "q",
        "\u051d": "w", "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M", "\u041d": "H",
        "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0425": "X", "\u03bf": "o", "\u03b1": "a",
        "\u03b5": "e", "\u03b9": "i", "\u03bd": "v", "\u039f": "O", "\u0399": "I", "\u0391": "A", "\u0395": "E",
    }
)  # fmt: skip


def _is_variation_selector(cp: int) -> bool:
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


def _selector_byte(cp: int) -> int:
    return cp - 0xFE00 if cp <= 0xFE0F else cp - 0xE0100 + 16


@dataclass(frozen=True)
class HiddenText:
    """Result of :func:`reveal_hidden`: the visible text plus whatever was hidden in it."""

    cleaned: str
    tag_payload: str
    selector_payload: str
    zero_width: int
    bidi_controls: int

    @property
    def found(self) -> bool:
        return bool(self.tag_payload or self.selector_payload or self.zero_width or self.bidi_controls)


def reveal_hidden(text: str) -> HiddenText:
    """Strip invisible characters and decode any payload they carry."""
    out: list[str] = []
    tags: list[str] = []
    selector_bytes: list[bytes] = []
    zero_width = bidi = 0
    i, n = 0, len(text)
    while i < n:
        cp = ord(text[i])
        if TAG_FIRST <= cp <= TAG_LAST:
            j = i
            while j < n and TAG_FIRST <= ord(text[j]) <= TAG_LAST:
                j += 1
            run = text[i:j]
            decoded = "".join(chr(ord(c) - TAG_FIRST) for c in run if 0x20 <= ord(c) - TAG_FIRST <= 0x7E)
            after_flag = i > 0 and ord(text[i - 1]) == BLACK_FLAG
            is_flag = after_flag and ord(run[-1]) == TAG_CANCEL and bool(_FLAG_TAGS.fullmatch(decoded))
            if not is_flag and decoded.strip():
                tags.append(decoded)
            i = j
            continue
        if _is_variation_selector(cp):
            j = i
            while j < n and _is_variation_selector(ord(text[j])):
                j += 1
            if j - i == 1 and cp <= 0xFE0F:
                out.append(text[i])  # a single presentation selector after an emoji is ordinary
            else:
                selector_bytes.append(bytes(_selector_byte(ord(c)) for c in text[i:j]))
            i = j
            continue
        if cp in ZERO_WIDTH:
            prev = text[i - 1] if i else ""
            nxt = text[i + 1] if i + 1 < n else ""
            legitimate = (cp in JOINERS and (not prev.isascii() or not nxt.isascii())) or (cp == 0xFEFF and i == 0)
            if legitimate:
                out.append(text[i])
            else:
                zero_width += 1
            i += 1
            continue
        if cp in BIDI_CONTROLS:
            bidi += 1
            i += 1
            continue
        out.append(text[i])
        i += 1
    selector_text = " ".join(b.decode("utf-8", errors="ignore") for b in selector_bytes).strip()
    return HiddenText(
        cleaned="".join(out),
        tag_payload=" ".join(tags).strip(),
        selector_payload=selector_text,
        zero_width=zero_width,
        bidi_controls=bidi,
    )


def normalise_for_matching(text: str) -> str:
    """Canonical view for rule matching: NFKC (folds full-width letters), common homoglyphs mapped to Latin,
    remaining zero-width characters dropped, datamarks read as spaces and horizontal whitespace
    collapsed."""
    folded = unicodedata.normalize("NFKC", text.replace(DATAMARK, " ")).translate(_CONFUSABLES)
    folded = "".join(ch for ch in folded if ord(ch) not in ZERO_WIDTH)
    return re.sub(r"[^\S\n]+", " ", folded)
