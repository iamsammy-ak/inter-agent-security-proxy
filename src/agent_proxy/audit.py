"""Tamper-evident audit log: hash-chained JSON Lines.

Each record stores the hash of the previous record, and its own hash covers every field including that link:

    hash_n = SHA-256(canonical_json(record_n without "hash" and "mac"))      record_n["prev"] = hash_(n-1)

Editing, deleting, inserting or reordering a record breaks the chain at that point, and :func:`verify_audit_log`
reports where. Two gaps a plain chain cannot close are handled explicitly:

* someone who can rewrite the whole file can recompute every hash, so an optional key adds an HMAC per record
  (``AGENT_PROXY_AUDIT_KEY``); without the key the chain cannot be forged;
* truncating the newest records leaves a valid shorter chain, so callers can pin the last known head hash
  (``expected_head``) from an external system.

Records hold tokens, rule ids and decisions, never raw PII values.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

GENESIS = "0" * 64


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def record_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class AuditRecord(BaseModel):
    """One audit entry."""

    seq: int
    ts: str
    event: str
    actor: str | None = None
    session: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    prev: str
    hash: str
    mac: str | None = None

    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude={"hash", "mac"})


class VerifyResult(BaseModel):
    """Outcome of a chain verification."""

    ok: bool
    records: int
    head: str
    error_seq: int | None = None
    error: str | None = None


class AuditLog:
    """Append-only, hash-chained log. ``path=None`` keeps records in memory (tests and the demo)."""

    def __init__(self, path: str | Path | None = None, *, key: bytes | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._key = key.encode("utf-8") if isinstance(key, str) else key
        self._memory: list[str] = []
        self._lock = threading.Lock()
        self._seq, self._head = 0, GENESIS
        if self.path is not None and self.path.exists():
            last = _last_line(self.path)
            if last:
                record = AuditRecord.model_validate_json(last)
                self._seq, self._head = record.seq, record.hash

    @property
    def head(self) -> str:
        """Hash of the newest record (``GENESIS`` when empty)."""
        return self._head

    def append(self, event: str, *, actor: str | None = None, session: str | None = None, **data: Any) -> AuditRecord:
        """Write one record and return it."""
        with self._lock:
            body = {
                "seq": self._seq + 1,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event": event,
                "actor": actor,
                "session": session,
                "data": json.loads(canonical_json(data)),
                "prev": self._head,
            }
            digest = record_hash(body)
            mac = hmac.new(self._key, digest.encode("ascii"), hashlib.sha256).hexdigest() if self._key else None
            record = AuditRecord(**body, hash=digest, mac=mac)
            line = record.model_dump_json(exclude_none=False)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line + "\n")
            else:
                self._memory.append(line)
            self._seq, self._head = record.seq, digest
            return record

    def lines(self) -> list[str]:
        """Raw JSON lines in order."""
        if self.path is None:
            return list(self._memory)
        return self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []

    def records(self) -> list[AuditRecord]:
        return [AuditRecord.model_validate_json(line) for line in self.lines() if line.strip()]

    def verify(self, *, expected_head: str | None = None) -> VerifyResult:
        return verify_lines(self.lines(), key=self._key, expected_head=expected_head)


def _last_line(path: Path) -> str | None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else None


def verify_lines(lines: list[str], *, key: bytes | str | None = None, expected_head: str | None = None) -> VerifyResult:
    """Check sequence numbers, hash links, record hashes and (with ``key``) the HMAC of every record."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    prev, count = GENESIS, 0
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = AuditRecord.model_validate_json(line)
        except ValueError as exc:
            return VerifyResult(ok=False, records=count, head=prev, error_seq=index, error=f"unparseable line: {exc}")
        if record.seq != count + 1:
            return VerifyResult(ok=False, records=count, head=prev, error_seq=record.seq,
                                error=f"sequence jumps from {count} to {record.seq} (record missing or inserted)")
        if record.prev != prev:
            return VerifyResult(ok=False, records=count, head=prev, error_seq=record.seq,
                                error="previous-hash link broken (record removed, inserted or reordered)")
        if record_hash(record.body()) != record.hash:
            return VerifyResult(ok=False, records=count, head=prev, error_seq=record.seq,
                                error="record content does not match its hash (edited)")
        if key_bytes is not None:
            expected_mac = hmac.new(key_bytes, record.hash.encode("ascii"), hashlib.sha256).hexdigest()
            if record.mac is None or not hmac.compare_digest(expected_mac, record.mac):
                return VerifyResult(ok=False, records=count, head=prev, error_seq=record.seq,
                                    error="HMAC does not match (chain recomputed without the key)")
        prev, count = record.hash, count + 1
    if expected_head is not None and prev != expected_head:
        return VerifyResult(ok=False, records=count, head=prev, error="head differs from the pinned hash (truncated?)")
    return VerifyResult(ok=True, records=count, head=prev)


def verify_audit_log(
    path: str | Path, *, key: bytes | str | None = None, expected_head: str | None = None
) -> VerifyResult:
    """Verify a log file on disk."""
    return verify_lines(Path(path).read_text(encoding="utf-8").splitlines(), key=key, expected_head=expected_head)
