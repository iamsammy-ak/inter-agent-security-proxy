"""Optional person-name detection with spaCy's ``en_core_web_sm`` named-entity recogniser.

Names have no checksum, so this is the least precise detector; it is off unless explicitly enabled.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..models import PIIType
from .detectors import PIISpan


def spacy_model_available(model: str = "en_core_web_sm") -> bool:
    """True when spaCy and the model package are installed."""
    try:
        import spacy.util
    except ImportError:
        return False
    return bool(spacy.util.is_package(model))


class SpacyNameFinder:
    """Callable finder yielding ``PERSON`` spans; the pipeline is loaded on first use."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.model = model
        self._nlp: Any = None

    def _load(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.model, disable=["parser", "lemmatizer", "attribute_ruler", "tagger"])
        return self._nlp

    def __call__(self, text: str) -> Iterator[PIISpan]:
        doc = self._load()(text)
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            # Entities can run across a line break ("Giulia Rossi" + newline + "Email"): keep the first line.
            first_line = ent.text.split("\n", 1)[0]
            start = ent.start_char + (len(first_line) - len(first_line.lstrip()))
            name = first_line.strip()
            if any(ch.isalpha() for ch in name):
                yield PIISpan(PIIType.PERSON, start, start + len(name), name, "person:spacy")
