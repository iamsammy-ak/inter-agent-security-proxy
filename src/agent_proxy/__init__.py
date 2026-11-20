"""Inter-agent security proxy: PII tokenisation, prompt-injection defence, least-privilege tool policy and a
tamper-evident audit log for messages passed between AI agents."""

from __future__ import annotations

__version__ = "0.1.0"

from .audit import AuditLog, verify_audit_log
from .injection import InjectionClassifier, InjectionDetector, InjectionVerdict, Spotlighter, load_classifier
from .models import PermissionDenied, PIIType, Principal
from .pii import Masker, PIIDetector, TokenVault
from .policy import PolicyEngine, PolicyFile, ToolCall, ToolDecision, load_policy
from .proxy import InjectionBlocked, PolicyViolation, RelayRequest, RelayResult, SecurityProxy

__all__ = [
    "AuditLog",
    "InjectionBlocked",
    "InjectionClassifier",
    "InjectionDetector",
    "InjectionVerdict",
    "Masker",
    "PIIDetector",
    "PIIType",
    "PermissionDenied",
    "PolicyEngine",
    "PolicyFile",
    "PolicyViolation",
    "Principal",
    "RelayRequest",
    "RelayResult",
    "SecurityProxy",
    "Spotlighter",
    "TokenVault",
    "ToolCall",
    "ToolDecision",
    "__version__",
    "load_classifier",
    "load_policy",
    "verify_audit_log",
]
