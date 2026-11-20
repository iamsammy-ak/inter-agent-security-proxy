"""The security proxy: one object that screens every message, tool result and tool call passing between agents.

Order of operations for untrusted content (a tool result or another agent's message):

1. **Detect** prompt injection on the raw text (rules, hidden characters, decoded payloads, links, classifier).
2. **Clean** invisible characters, then **mask** PII with session-stable tokens.
3. **Act** on the agent's ``on_injection`` setting: ``sanitize`` (spotlight and pass on), ``quarantine`` (withhold
   the content, keep a notice) or ``block`` (refuse the request).
4. **Spotlight**: wrap the masked text in provenance delimiters with datamarking and defang exfiltration links.
5. **Enforce** least privilege on any tool calls the model proposes, and **audit** every decision.

Model contexts only ever see tokens. Values come back only at the tool-execution boundary, and only for
principals holding ``pii:reveal`` and the matching data class.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .audit import AuditLog
from .config import Settings, load_settings
from .injection.classifier import InjectionClassifier, load_classifier
from .injection.detector import InjectionDetector, InjectionVerdict
from .injection.spotlight import Spotlighter, SpotlightMode, defang_links, system_notice, unwrap
from .injection.unicode import reveal_hidden
from .llm import Upstream, message_text
from .models import PermissionDenied, Principal
from .pii.detectors import PIIDetector
from .pii.masker import MaskedEntity, Masker
from .pii.names import SpacyNameFinder
from .pii.vault import TokenVault
from .policy import InjectionAction, PolicyEngine, PolicyFile, ToolCall, ToolDecision, load_policy

Action = Literal["pass", "sanitize", "quarantine", "block"]


class ProxyError(Exception):
    """Base class for refusals the HTTP layer turns into 4xx responses."""

    status_code = 400


class PolicyViolation(ProxyError):
    """The request breaks the caller's policy (unknown agent, relay not allowed, message too long)."""

    status_code = 403

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class InjectionBlocked(ProxyError):
    """Content was classified as an injection and the agent's policy says ``block``."""

    status_code = 422

    def __init__(self, verdict: InjectionVerdict) -> None:
        super().__init__(f"prompt injection detected ({', '.join(verdict.categories()) or 'classifier'})")
        self.verdict = verdict


class Screened(BaseModel):
    """Untrusted text after screening. ``text`` is what the next agent receives (``None`` when blocked)."""

    text: str | None
    action: Action
    verdict: InjectionVerdict
    pii: list[MaskedEntity]
    removed_invisible: int = 0
    defanged_urls: list[str] = Field(default_factory=list)


class InspectReport(BaseModel):
    """Read-only analysis of a text: masked copy, PII found (types and tokens only) and the injection verdict."""

    masked_text: str
    pii: list[MaskedEntity]
    pii_counts: dict[str, int]
    verdict: InjectionVerdict


class RelayRequest(BaseModel):
    """A message or tool result one agent hands to another."""

    session_id: str
    sender: str
    recipient: str
    kind: Literal["message", "tool_result"] = "message"
    tool_name: str | None = None
    content: str


class RelayResult(BaseModel):
    """What the recipient gets, plus the evidence behind it."""

    delivered: bool
    content: str | None
    action: Action
    verdict: InjectionVerdict
    pii: list[MaskedEntity]
    system_notice: str | None = None
    reasons: list[str] = Field(default_factory=list)
    audit_seq: int


class SecurityProxy:
    """In-process middleware. The FastAPI service in :mod:`agent_proxy.server` is a thin layer over it."""

    def __init__(
        self,
        policy: PolicyFile,
        *,
        vault: TokenVault | None = None,
        classifier: InjectionClassifier | None = None,
        audit: AuditLog | None = None,
        spotlight_mode: SpotlightMode = "datamark",
        pii_detector: PIIDetector | None = None,
    ) -> None:
        self.policy = policy
        self.vault = vault or TokenVault()
        self.masker = Masker(self.vault, pii_detector)
        self.engine = PolicyEngine(policy, self.vault)
        allowlist = tuple(policy.link_allowlist)
        self.detector = InjectionDetector(classifier, link_allowlist=allowlist)
        self.spotlighter = Spotlighter(spotlight_mode, link_allowlist=allowlist)
        self.audit = audit or AuditLog()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        policy_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        names: bool = False,
    ) -> SecurityProxy:
        """Build a proxy from environment settings (policy file, audit path and keys, classifier choice)."""
        settings = settings or load_settings()
        path = policy_path or settings.policy_path
        policy = load_policy(path) if path else PolicyFile()
        detector = PIIDetector(extra_finders=[SpacyNameFinder()] if names else ())
        return cls(
            policy,
            vault=TokenVault(settings.secret),
            classifier=load_classifier(settings.classifier),
            audit=AuditLog(audit_path or settings.audit_path, key=settings.audit_key),
            pii_detector=detector,
        )

    # ------------------------------------------------------------------ screening
    def _detect(self, text: str, session_id: str) -> InjectionVerdict:
        """Run detection on a masked copy, so excerpts in verdicts and audit records carry tokens, not values.

        Masking leaves invisible characters in place, so hidden payloads are still found; excerpts taken from
        decoded payloads are masked separately."""
        verdict = self.detector.inspect(self.masker.mask(text, session_id).text)
        hits = [
            h if h.view == "text" else h.model_copy(update={"excerpt": self.masker.mask(h.excerpt, session_id).text})
            for h in verdict.hits
        ]
        return verdict.model_copy(update={"hits": hits})

    def inspect(self, text: str, *, session_id: str = "inspect", actor: str | None = None) -> InspectReport:
        """Analyse without forwarding anything. Audited as ``inspect``."""
        verdict = self._detect(text, session_id)
        masked = self.masker.mask(reveal_hidden(text).cleaned, session_id)
        report = InspectReport(masked_text=masked.text, pii=masked.entities, pii_counts=masked.counts(),
                               verdict=verdict)
        self.audit.append("inspect", actor=actor, session=session_id, pii=report.pii_counts,
                          injection=verdict.is_injection, score=verdict.score, rules=[h.rule_id for h in verdict.hits])
        return report

    def screen(
        self,
        text: str,
        *,
        session_id: str,
        source: str,
        kind: str,
        tool: str | None = None,
        on_injection: InjectionAction = "sanitize",
        spotlight: bool = True,
    ) -> Screened:
        """Detect, clean, mask and (optionally) spotlight one piece of untrusted text."""
        inner = unwrap(text)
        already_wrapped = inner is not None
        verdict = self._detect(inner if inner is not None else text, session_id)
        action: Action = on_injection if verdict.is_injection else "pass"
        if action == "block":
            return Screened(text=None, action=action, verdict=verdict, pii=[])
        hidden = reveal_hidden(text)
        masked = self.masker.mask(hidden.cleaned, session_id)
        removed = len(text) - len(hidden.cleaned)
        if action == "quarantine":
            notice = (f"[agent-proxy withheld {kind} from {source}: prompt injection detected "
                      f"({', '.join(verdict.categories()) or 'classifier'}); {len(masked.entities)} PII value(s) "
                      "were tokenised]")
            return Screened(text=notice, action=action, verdict=verdict, pii=masked.entities, removed_invisible=removed)
        if already_wrapped:  # produced by this proxy earlier in the workflow: tokens are stable, do not wrap twice
            return Screened(text=masked.text, action=action, verdict=verdict, pii=masked.entities)
        if spotlight or verdict.is_injection:
            wrapped = self.spotlighter.wrap(masked.text, source=source, kind=kind, tool=tool)
            return Screened(text=wrapped.text, action=action, verdict=verdict, pii=masked.entities,
                            removed_invisible=removed, defanged_urls=wrapped.defanged_urls)
        return Screened(text=masked.text, action=action, verdict=verdict, pii=masked.entities,
                        removed_invisible=removed)

    # ------------------------------------------------------------------ relay
    def relay(self, request: RelayRequest) -> RelayResult:
        """Screen a message or tool result from ``sender`` before ``recipient`` sees it."""
        reasons = self.engine.check_relay(request.sender, request.recipient)
        recipient = self.engine.agent(request.recipient)
        if recipient and len(request.content) > recipient.limits.max_message_chars:
            reasons.append(f"content exceeds {recipient.limits.max_message_chars} characters")
        if reasons:
            record = self.audit.append("relay.denied", actor=request.sender, session=request.session_id,
                                       recipient=request.recipient, reasons=reasons)
            empty = InjectionVerdict(is_injection=False, score=0.0, rule_score=0.0)
            return RelayResult(delivered=False, content=None, action="block", verdict=empty, pii=[], reasons=reasons,
                               audit_seq=record.seq)
        assert recipient is not None
        screened = self.screen(request.content, session_id=request.session_id, source=request.sender,
                               kind=request.kind, tool=request.tool_name, on_injection=recipient.on_injection)
        delivered = screened.text is not None
        record = self.audit.append(
            "relay" if delivered else "relay.blocked",
            actor=request.sender, session=request.session_id, recipient=request.recipient, kind=request.kind,
            tool=request.tool_name, action=screened.action, injection=screened.verdict.is_injection,
            score=screened.verdict.score, rules=sorted({h.rule_id for h in screened.verdict.hits}),
            pii=[e.token for e in screened.pii], defanged=len(screened.defanged_urls),
            invisible_removed=screened.removed_invisible,
        )
        return RelayResult(delivered=delivered, content=screened.text, action=screened.action,
                           verdict=screened.verdict, pii=screened.pii,
                           system_notice=system_notice(self.spotlighter.mode) if delivered else None,
                           reasons=[] if delivered else ["prompt injection detected; recipient policy is block"],
                           audit_seq=record.seq)

    # ------------------------------------------------------------------ tool calls and reveal
    def check_tool_calls(self, session_id: str, agent_id: str, calls: list[ToolCall]) -> list[ToolDecision]:
        """Policy decision per proposed call; each decision is audited (arguments as tokens only)."""
        decisions = self.engine.check_tool_calls(session_id, agent_id, calls)
        for call, decision in zip(calls, decisions, strict=True):
            self.audit.append("tool_call.allowed" if decision.allowed else "tool_call.denied", actor=agent_id,
                              session=session_id, tool=call.name, call_id=call.id, reasons=decision.reasons,
                              arguments=self.masker.mask(json.dumps(call.arguments, sort_keys=True), session_id).text)
        return decisions

    def reveal(self, session_id: str, text: str, principal: Principal) -> str:
        """Rehydrate tokens for a principal holding ``pii:reveal`` (limited to its data classes)."""
        try:
            revealed = self.vault.rehydrate(session_id, text, principal)
        except PermissionDenied:
            self.audit.append("pii.reveal.denied", actor=principal.agent_id, session=session_id)
            raise
        self.audit.append("pii.reveal", actor=principal.agent_id, session=session_id,
                          classes=sorted(c.value for c in principal.revealable()))
        return revealed

    # ------------------------------------------------------------------ chat completions
    def guard_request(
        self, agent_id: str, session_id: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], list[Screened]]:
        """Screen an OpenAI chat request from ``agent_id``. Tool results are spotlighted; every message is masked."""
        policy = self.engine.agent(agent_id)
        if policy is None:
            raise PolicyViolation([f"unknown agent {agent_id!r}: default deny"])
        if payload.get("stream"):
            raise PolicyViolation(["streaming responses are not supported by the proxy"])
        body = copy.deepcopy(payload)
        screened: list[Screened] = []
        spotlighted = False
        for message in body.get("messages") or []:
            role = message.get("role")
            text = message_text(message)
            if len(text) > policy.limits.max_message_chars:
                raise PolicyViolation([f"a {role} message exceeds {policy.limits.max_message_chars} characters"])
            if role in ("user", "tool"):
                result = self.screen(text, session_id=session_id, source=str(message.get("name") or role),
                                     kind="tool_result" if role == "tool" else "message",
                                     on_injection=policy.on_injection, spotlight=role == "tool")
                if result.action == "block":
                    raise InjectionBlocked(result.verdict)
                screened.append(result)
                spotlighted = spotlighted or (result.text or "").startswith("«untrusted")
                message["content"] = result.text
            elif isinstance(message.get("content"), str):
                message["content"] = self.masker.mask(message["content"], session_id).text
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = self.masker.mask(function["arguments"], session_id).text
        if spotlighted:
            notice = system_notice(self.spotlighter.mode)
            messages = body.setdefault("messages", [])
            if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                messages[0]["content"] = f"{messages[0]['content']}\n\n{notice}"
            else:
                messages.insert(0, {"role": "system", "content": notice})
        return body, screened

    def guard_response(
        self, agent_id: str, session_id: str, response: dict[str, Any]
    ) -> tuple[dict[str, Any], list[ToolDecision]]:
        """Mask PII and defang links in the reply, and drop tool calls the policy refuses."""
        body = copy.deepcopy(response)
        all_decisions: list[ToolDecision] = []
        for choice in body.get("choices") or []:
            message = choice.get("message") or {}
            if isinstance(message.get("content"), str):
                masked = self.masker.mask(message["content"], session_id).text
                message["content"], _ = defang_links(masked, tuple(self.policy.link_allowlist))
            raw_calls = message.get("tool_calls") or []
            if not raw_calls:
                continue
            calls = [ToolCall.from_openai(c) for c in raw_calls]
            decisions = self.check_tool_calls(session_id, agent_id, calls)
            all_decisions.extend(decisions)
            kept = []
            for raw, decision in zip(raw_calls, decisions, strict=True):
                if decision.allowed:
                    raw = copy.deepcopy(raw)
                    raw.setdefault("function", {})["arguments"] = json.dumps(decision.execute_arguments)
                    kept.append(raw)
            denied = [d for d in decisions if not d.allowed]
            if kept:
                message["tool_calls"] = kept
            else:
                message.pop("tool_calls", None)
                choice["finish_reason"] = "stop"
            if denied:
                note = "[agent-proxy blocked " + "; ".join(f"{d.tool}: {d.reasons[0]}" for d in denied) + "]"
                message["content"] = f"{message.get('content') or ''}\n{note}".strip()
        body["agent_proxy"] = {"session_id": session_id,
                               "tool_decisions": [d.model_dump(exclude={"execute_arguments"}) for d in all_decisions]}
        return body, all_decisions

    def chat(self, agent_id: str, session_id: str, payload: dict[str, Any], upstream: Upstream) -> dict[str, Any]:
        """Full pipeline: guard the request, call the upstream model, guard the response, audit both ends."""
        try:
            request, screened = self.guard_request(agent_id, session_id, payload)
        except InjectionBlocked as exc:
            self.audit.append("chat.blocked", actor=agent_id, session=session_id, score=exc.verdict.score,
                              rules=sorted({h.rule_id for h in exc.verdict.hits}))
            raise
        self.audit.append("chat.request", actor=agent_id, session=session_id, upstream=upstream.name,
                          messages=len(request.get("messages") or []),
                          injections=sum(s.verdict.is_injection for s in screened),
                          pii=sorted({e.token for s in screened for e in s.pii}))
        response, decisions = self.guard_response(agent_id, session_id, upstream.create(request))
        self.audit.append("chat.response", actor=agent_id, session=session_id,
                          allowed=sum(d.allowed for d in decisions), denied=sum(not d.allowed for d in decisions))
        return response
