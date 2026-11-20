"""Per-session token vault: deterministic HMAC tokens out, scoped rehydration back in.

Tokens look like ``<EMAIL_7f3a9c1e>``. The hex part is ``HMAC-SHA256(session_key, type || canonical value)``, where
``session_key = HMAC-SHA256(secret, session_id)``. Consequences:

* the same value always maps to the same token within a session, even if it is formatted differently
  (``IT60 X054 ...`` and ``IT60X054...``), so multi-turn conversations stay coherent;
* the same value maps to unrelated tokens in different sessions, so tokens cannot be joined across sessions;
* a token cannot be forged or reversed without the secret, and only tokens the vault issued for that session can
  be rehydrated.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from ..models import SCOPE_PII_REVEAL, PermissionDenied, PIIType, Principal
from .detectors import TOKEN_RE, normalise_date


def canonical(pii_type: PIIType, value: str) -> str:
    """Normal form used for hashing, so formatting differences do not create new tokens."""
    v = value.strip()
    if pii_type is PIIType.EMAIL:
        return v.lower()
    if pii_type is PIIType.PHONE:
        digits = re.sub(r"\D", "", v)
        if v.startswith("+"):
            return "+" + digits
        if v.startswith("00"):
            return "+" + digits[2:]
        if digits[:1] in ("0", "3") and 8 <= len(digits) <= 11:
            return "+39" + digits  # Italian national number
        if len(digits) == 10:
            return "+1" + digits  # North American number
        return digits
    if pii_type in (PIIType.IBAN, PIIType.CODICE_FISCALE):
        return re.sub(r"\s+", "", v).upper()
    if pii_type is PIIType.CARD:
        return re.sub(r"\D", "", v)
    if pii_type is PIIType.IP_ADDRESS:
        try:
            return ipaddress.ip_address(v.split("%", 1)[0]).compressed
        except ValueError:
            return v.lower()
    if pii_type is PIIType.DATE_OF_BIRTH:
        return normalise_date(v)
    if pii_type is PIIType.PERSON:
        return " ".join(v.split()).casefold()
    return v


@dataclass(frozen=True)
class VaultEntry:
    """What a token stands for. ``value`` is the first-seen original spelling, used for rehydration."""

    type: PIIType
    canonical: str
    value: str


@dataclass
class _Session:
    key: bytes
    by_token: dict[str, VaultEntry] = field(default_factory=dict)
    by_value: dict[tuple[PIIType, str], str] = field(default_factory=dict)


class TokenVault:
    """Thread-safe in-memory store of token mappings, bounded to ``max_sessions`` (least recently used evicted)."""

    def __init__(self, secret: bytes | str | None = None, *, token_hex: int = 8, max_sessions: int = 10_000) -> None:
        if not 4 <= token_hex <= 64:
            raise ValueError("token_hex must be between 4 and 64")
        if secret is None:
            secret = secrets.token_bytes(32)
        self._secret = secret.encode("utf-8") if isinstance(secret, str) else secret
        self.token_hex = token_hex
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def _session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            key = hmac.new(self._secret, b"agent-proxy/session/" + session_id.encode("utf-8"), hashlib.sha256)
            session = _Session(key=key.digest())
            self._sessions[session_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return session

    def tokenize(self, session_id: str, pii_type: PIIType, value: str) -> str:
        """Token for ``value``; issues and stores a new one the first time the value is seen in the session."""
        canon = canonical(pii_type, value)
        with self._lock:
            session = self._session(session_id)
            existing = session.by_value.get((pii_type, canon))
            if existing is not None:
                return existing
            message = pii_type.value.encode("utf-8") + b"\x00" + canon.encode("utf-8")
            digest = hmac.new(session.key, message, hashlib.sha256).hexdigest()
            length = self.token_hex
            token = f"<{pii_type.value}_{digest[:length]}>"
            while token in session.by_token:  # truncated-HMAC collision with a different value: lengthen
                length = min(64, length * 2)
                token = f"<{pii_type.value}_{digest[:length]}>"
            session.by_token[token] = VaultEntry(pii_type, canon, value)
            session.by_value[(pii_type, canon)] = token
            return token

    def resolve(self, session_id: str, token: str) -> VaultEntry | None:
        """Internal lookup used by policy checks (for example, where a tokenised e-mail would really go).

        It performs no scope check, so its result must never be returned to a caller; use :meth:`rehydrate`.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            return session.by_token.get(token) if session else None

    def rehydrate(self, session_id: str, text: str, principal: Principal) -> str:
        """Replace tokens with their values, limited to the classes ``principal`` may see.

        Raises :class:`PermissionDenied` when the principal lacks the ``pii:reveal`` scope. Tokens of other classes,
        and strings that only look like tokens, are left untouched.
        """
        if not principal.has_scope(SCOPE_PII_REVEAL):
            raise PermissionDenied(f"{principal.agent_id} lacks the {SCOPE_PII_REVEAL} scope")
        allowed = principal.revealable()

        def swap(match: re.Match[str]) -> str:
            entry = self.resolve(session_id, match.group())
            return entry.value if entry is not None and entry.type in allowed else match.group()

        return TOKEN_RE.sub(swap, text)

    def size(self, session_id: str) -> int:
        """Number of distinct values tokenised in a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            return len(session.by_token) if session else 0

    def drop_session(self, session_id: str) -> None:
        """Forget every mapping of a session (data minimisation at the end of a workflow)."""
        with self._lock:
            self._sessions.pop(session_id, None)
