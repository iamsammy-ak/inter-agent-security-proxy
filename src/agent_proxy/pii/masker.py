"""Replace detected PII with session-stable tokens."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from ..models import PIIType
from .detectors import PIIDetector
from .vault import TokenVault


class MaskedEntity(BaseModel):
    """One replaced value. Positions refer to the original text; the value itself is never included."""

    type: PIIType
    token: str
    start: int
    end: int
    detector: str


class MaskResult(BaseModel):
    """Masked text plus what was replaced."""

    text: str
    entities: list[MaskedEntity]

    def counts(self) -> dict[str, int]:
        return dict(Counter(e.type.value for e in self.entities))


class Masker:
    """Detects PII and swaps each value for its vault token. Masking is idempotent: tokens are never re-masked."""

    def __init__(self, vault: TokenVault, detector: PIIDetector | None = None) -> None:
        self.vault = vault
        self.detector = detector or PIIDetector()

    def mask(self, text: str, session_id: str) -> MaskResult:
        spans = self.detector.detect(text)
        pieces: list[str] = []
        entities: list[MaskedEntity] = []
        cursor = 0
        for span in spans:
            token = self.vault.tokenize(session_id, span.type, span.value)
            pieces.append(text[cursor : span.start])
            pieces.append(token)
            entities.append(MaskedEntity(type=span.type, token=token, start=span.start, end=span.end,
                                         detector=span.detector))
            cursor = span.end
        pieces.append(text[cursor:])
        return MaskResult(text="".join(pieces), entities=entities)
