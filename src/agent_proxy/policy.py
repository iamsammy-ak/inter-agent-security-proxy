"""Least-privilege policy: what each agent may call, see, contact and send.

Policies are YAML files validated by Pydantic. Anything not explicitly allowed is denied: an unknown agent has
no tools, a tool missing from an agent's list is refused, and every e-mail address or URL found in tool
arguments must match the agent's destination allowlist. Tokenised arguments are resolved internally for these
checks, so ``send_email(to="<EMAIL_1a2b3c4d>")`` is judged by where it would really go.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .models import SCOPE_PII_REVEAL, PIIType, Principal
from .pii.detectors import TOKEN_RE
from .pii.vault import TokenVault

InjectionAction = Literal["sanitize", "quarantine", "block"]

_URL = re.compile(r"https?://[^\s\"'<>)]+", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


class ArgConstraint(BaseModel):
    """Constraint on one tool argument."""

    min: float | None = None
    max: float | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    max_length: int | None = None
    destination: bool = Field(False, description="the value is a host, URL or e-mail checked against destinations")


class ToolRule(BaseModel):
    """A tool an agent may call, with optional argument constraints and a per-session budget."""

    name: str
    args: dict[str, ArgConstraint] = Field(default_factory=dict)
    max_calls_per_session: int | None = None


class Limits(BaseModel):
    max_tool_calls_per_response: int = 4
    max_message_chars: int = 50_000


class AgentPolicy(BaseModel):
    """Everything one agent is allowed to do."""

    description: str = ""
    key_sha256: str | None = Field(None, description="SHA-256 of the agent's bearer key; omit to trust the header")
    scopes: list[str] = Field(default_factory=list)
    data_classes: list[PIIType] = Field(default_factory=list)
    tools: list[ToolRule] = Field(default_factory=list)
    destinations: list[str] = Field(default_factory=list)
    can_send_to: list[str] = Field(default_factory=list)
    on_injection: InjectionAction = "sanitize"
    limits: Limits = Field(default_factory=Limits)

    def tool(self, name: str) -> ToolRule | None:
        return next((t for t in self.tools if t.name == name), None)


class PolicyFile(BaseModel):
    """Top-level policy document."""

    version: int = 1
    link_allowlist: list[str] = Field(default_factory=list)
    agents: dict[str, AgentPolicy] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError("only policy version 1 is supported")
        return value


def load_policy(path: str | Path) -> PolicyFile:
    """Read and validate a YAML policy file."""
    return PolicyFile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


class ToolCall(BaseModel):
    """A tool call proposed by a model (OpenAI ``tool_calls`` shape, arguments already parsed)."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    malformed: bool = False

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> ToolCall:
        function = raw.get("function") or {}
        text = function.get("arguments") or "{}"
        try:
            args = json.loads(text) if isinstance(text, str) else dict(text)
            if not isinstance(args, dict):
                raise ValueError("arguments are not an object")
        except (ValueError, TypeError):
            return cls(id=str(raw.get("id", "")), name=str(function.get("name", "")), malformed=True)
        return cls(id=str(raw.get("id", "")), name=str(function.get("name", "")), arguments=args)


class ToolDecision(BaseModel):
    """Verdict for one proposed call. ``execute_arguments`` is set only for allowed calls: tokens of classes the
    agent may see are rehydrated (when it holds ``pii:reveal``) so the tool receives real values."""

    call_id: str
    tool: str
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    execute_arguments: dict[str, Any] | None = None


def destination_allowed(host: str, allowlist: list[str]) -> bool:
    host = host.lower().rstrip(".")
    for entry in allowlist:
        entry = entry.lower()
        if entry.startswith("*."):
            if host.endswith(entry[1:]):
                return True
        elif host == entry:
            return True
    return False


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def _hosts_in(text: str) -> list[str]:
    hosts = [re.sub(r"^https?://", "", u, flags=re.I).split("/", 1)[0].split("?", 1)[0].split(":", 1)[0]
             for u in _URL.findall(text)]
    hosts += _EMAIL.findall(text)
    return hosts


class PolicyEngine:
    """Evaluates tool calls and relays against a :class:`PolicyFile`. Keeps per-session call counters."""

    def __init__(self, policy: PolicyFile, vault: TokenVault) -> None:
        self.policy = policy
        self.vault = vault
        self._calls: dict[tuple[str, str, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ identity
    def agent(self, agent_id: str) -> AgentPolicy | None:
        return self.policy.agents.get(agent_id)

    def principal(self, agent_id: str) -> Principal:
        """Principal for a known agent; unknown agents get no scopes and no data classes."""
        policy = self.agent(agent_id)
        if policy is None:
            return Principal(agent_id=agent_id)
        return Principal(agent_id=agent_id, scopes=frozenset(policy.scopes),
                         data_classes=frozenset(policy.data_classes))

    def authenticate(self, agent_id: str, bearer: str | None) -> Principal | None:
        """Principal if the agent exists and, when it has a key hash, the bearer key matches; otherwise ``None``."""
        policy = self.agent(agent_id)
        if policy is None:
            return None
        if policy.key_sha256:
            presented = hashlib.sha256((bearer or "").encode("utf-8")).hexdigest()
            if not hmac.compare_digest(presented, policy.key_sha256.lower()):
                return None
        return self.principal(agent_id)

    # ------------------------------------------------------------------ relay
    def check_relay(self, sender: str, recipient: str) -> list[str]:
        """Reasons the relay is refused (empty when allowed)."""
        reasons: list[str] = []
        sender_policy = self.agent(sender)
        if sender_policy is None:
            reasons.append(f"unknown sender {sender!r}")
        elif recipient not in sender_policy.can_send_to:
            reasons.append(f"{sender} may not send to {recipient}")
        if self.agent(recipient) is None:
            reasons.append(f"unknown recipient {recipient!r}")
        return reasons

    # ------------------------------------------------------------------ tool calls
    def _resolve_tokens(self, session_id: str, text: str) -> str:
        def swap(m: re.Match[str]) -> str:
            entry = self.vault.resolve(session_id, m.group())
            return entry.value if entry else m.group()

        return TOKEN_RE.sub(swap, text)

    def _check_arguments(self, session_id: str, rule: ToolRule, policy: AgentPolicy, args: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        reported: set[str] = set()
        for name, constraint in rule.args.items():
            if name not in args:
                continue
            value = args[name]
            if constraint.min is not None or constraint.max is not None:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    reasons.append(f"{name}={value!r} is not a number")
                    continue
                if constraint.max is not None and number > constraint.max:
                    reasons.append(f"{name}={number:g} exceeds the limit of {constraint.max:g}")
                if constraint.min is not None and number < constraint.min:
                    reasons.append(f"{name}={number:g} is below the minimum of {constraint.min:g}")
            if constraint.enum is not None and value not in constraint.enum:
                reasons.append(f"{name}={value!r} is not one of {constraint.enum}")
            if constraint.pattern is not None and not re.fullmatch(constraint.pattern, str(value)):
                reasons.append(f"{name} does not match the required pattern")
            if constraint.max_length is not None and len(str(value)) > constraint.max_length:
                reasons.append(f"{name} is longer than {constraint.max_length} characters")
            if constraint.destination:
                target = self._resolve_tokens(session_id, str(value))
                hosts = _hosts_in(target) or [target.strip().lower()]
                for host in hosts:
                    if not destination_allowed(host, policy.destinations):
                        reasons.append(f"{name} points to {host}, which is not an allowed destination")
                        reported.add(host)
        # Any URL or e-mail anywhere in the arguments must also be an allowed destination (exfiltration guard).
        for text in _strings(args):
            for host in _hosts_in(self._resolve_tokens(session_id, text)):
                if host not in reported and not destination_allowed(host, policy.destinations):
                    reasons.append(f"arguments reference {host}, which is not an allowed destination")
                    reported.add(host)
        return reasons

    def _rehydrate_args(self, session_id: str, principal: Principal, value: Any) -> Any:
        if isinstance(value, str):
            return self.vault.rehydrate(session_id, value, principal) if principal.revealable() else value
        if isinstance(value, dict):
            return {k: self._rehydrate_args(session_id, principal, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._rehydrate_args(session_id, principal, v) for v in value]
        return value

    def check_tool_calls(self, session_id: str, agent_id: str, calls: list[ToolCall]) -> list[ToolDecision]:
        """Allow or deny each call. Allowed calls count against their per-session budget."""
        policy = self.agent(agent_id)
        principal = self.principal(agent_id)
        decisions: list[ToolDecision] = []
        for index, call in enumerate(calls):
            reasons: list[str] = []
            rule = policy.tool(call.name) if policy else None
            if policy is None:
                reasons.append(f"unknown agent {agent_id!r}: default deny")
            elif call.malformed:
                reasons.append("tool arguments are not a valid JSON object")
            elif rule is None:
                reasons.append(f"{call.name} is not in {agent_id}'s tool allowlist")
            else:
                if index >= policy.limits.max_tool_calls_per_response:
                    reasons.append(f"more than {policy.limits.max_tool_calls_per_response} tool calls in one response")
                reasons.extend(self._check_arguments(session_id, rule, policy, call.arguments))
                if rule.max_calls_per_session is not None:
                    with self._lock:
                        used = self._calls[(session_id, agent_id, call.name)]
                    if used >= rule.max_calls_per_session:
                        reasons.append(f"{call.name} already used {used} time(s) in this session (limit "
                                       f"{rule.max_calls_per_session})")
            allowed = not reasons
            execute_arguments = None
            if allowed:
                with self._lock:
                    self._calls[(session_id, agent_id, call.name)] += 1
                if principal.has_scope(SCOPE_PII_REVEAL):
                    execute_arguments = self._rehydrate_args(session_id, principal, call.arguments)
                else:
                    execute_arguments = call.arguments
            decisions.append(ToolDecision(call_id=call.id, tool=call.name, allowed=allowed, reasons=reasons,
                                          execute_arguments=execute_arguments))
        return decisions
