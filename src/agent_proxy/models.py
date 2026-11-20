"""Shared data contracts: PII classes, principals and scopes."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

SCOPE_PII_REVEAL = "pii:reveal"
SCOPE_AUDIT_READ = "audit:read"


class PIIType(str, Enum):
    """Data classes the proxy recognises. The value is the prefix used in tokens such as ``<EMAIL_7f3a9c1e>``."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    IBAN = "IBAN"
    CARD = "CARD"
    CODICE_FISCALE = "CODICE_FISCALE"
    IP_ADDRESS = "IP"
    DATE_OF_BIRTH = "DOB"
    SECRET = "SECRET"
    PERSON = "PERSON"


class Principal(BaseModel):
    """An authenticated caller: an agent or a human-facing service.

    ``data_classes`` lists the PII classes the principal may receive in clear text, and only when it also holds the
    ``pii:reveal`` scope.
    """

    agent_id: str
    scopes: frozenset[str] = Field(default_factory=frozenset)
    data_classes: frozenset[PIIType] = Field(default_factory=frozenset)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def revealable(self) -> frozenset[PIIType]:
        """PII classes this principal may see in clear (empty without ``pii:reveal``)."""
        return self.data_classes if self.has_scope(SCOPE_PII_REVEAL) else frozenset()


class PermissionDenied(PermissionError):
    """Raised when a principal lacks the scope an operation needs."""
