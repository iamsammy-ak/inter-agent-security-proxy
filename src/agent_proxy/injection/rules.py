"""Hand-written injection rules plus structural checks for links, hidden text and encodings.

Every rule carries a weight in (0, 1]. Weights combine as a noisy-OR, ``1 - prod(1 - w)``, so one strong signal
(0.9) or two moderate ones (0.45 + 0.45 -> 0.70) cross the default threshold of 0.5, while a single weak signal
(a role marker such as ``System:`` in a log excerpt) does not. Patterns were written against the deepset *train*
split and general knowledge of published attacks; the test split was not used to write them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

_F = re.I | re.M


@dataclass(frozen=True)
class Rule:
    """One pattern-based signal."""

    id: str
    category: str
    weight: float
    pattern: re.Pattern[str]
    description: str


def _rule(rule_id: str, category: str, weight: float, pattern: str, description: str) -> Rule:
    return Rule(rule_id, category, weight, re.compile(pattern, _F), description)


_VERBS_EN = (
    r"(?:ignore|ignoring|disregard|disregarding|forget|forgetting|skip|override|overrule|bypass|neglect|drop|"
    r"abandon|discard|set aside|leave behind|remove)"
)
_PRIOR = (
    r"(?:previous|prior|preceding|above|earlier|foregoing|initial|original|old|all|any|your|the|these|those|system)"
)
_INSTR_EN = (
    r"(?:instructions?|directions?|directives?|prompts?|rules|guidelines|commands?|orders?|context|constraints?|"
    r"programming|tasks?|assignments?|information|documents?|articles?|everything)"
)
_GAP = r"(?:\W+\w+){0,4}?\W+"

RULES: tuple[Rule, ...] = (
    # --- instruction override -----------------------------------------------------------------------------
    _rule("OVR-EN-1", "instruction_override", 0.9,
          rf"\b{_VERBS_EN}\b{_GAP}{_PRIOR}\b(?:\W+\w+){{0,3}}?\W+{_INSTR_EN}\b",
          "Tells the model to ignore or forget its earlier instructions"),
    _rule("OVR-EN-2", "instruction_override", 0.8, r"\bforget\W+(?:about\W+)?(?:everything|all|anything)\b",
          "'Forget everything'"),
    _rule("OVR-EN-3", "instruction_override", 0.8, r"\bignore (?:the )?(?:above|previous|preceding)\b",
          "'Ignore the above'"),
    _rule("OVR-EN-4", "instruction_override", 0.8,
          r"\b(?:your|the) (?:new )?instructions (?:are|is) now\b|\bchange your (?:instructions|rules|programming)\b",
          "Rewrites the model's instructions"),
    _rule("OVR-EN-5", "instruction_override", 0.7,
          r"\b(?:regardless of|disregarding|despite|ignoring) (?:the |what |any )?(?:articles?|documents?|context|"
          r"you(?:'ve| have)? been told|you were told|instructions|sources)\b",
          "Asks the model to set aside its context"),
    _rule("OVR-EN-6", "instruction_override", 0.45,
          r"\b(?:new|updated|revised|additional|further|different) (?:instructions?|tasks?|assignments?)\b[^.\n]{0,20}"
          r"(?:follow|:|\s-)",
          "Announces new instructions"),
    _rule("OVR-EN-7", "instruction_override", 0.45,
          r"\b(?:do not|don'?t|dont|don'nt) (?:answer|look|use|follow|consider|read)\b[^.\n]{0,50}"
          r"\b(?:but|instead|just)\b",
          "'Do not answer ... but instead'"),
    _rule("OVR-EN-8", "instruction_override", 0.5,
          r"\binstead,? (?:just )?(?:say|print|output|write|respond with|reply with|answer with)\b",
          "Substitutes a different output"),
    _rule("OVR-DE-1", "instruction_override", 0.9,
          r"\b(?:ignorier\w*|vergiss|vergessen sie|missacht\w*|überspring\w*|verwirf)\b(?:\W+\w+){0,4}?\W+"
          r"(?:anweisung\w*|instruktion\w*|befehl\w*|vorgaben|regeln|aufgaben?|aufträge|auftrag|informationen|"
          r"angaben|alles|ausführungen)\b",
          "German: ignore or forget the instructions"),
    _rule("OVR-DE-2", "instruction_override", 0.8,
          r"\b(?:anweisungen|ausführungen|befehle|vorgaben|informationen)\b(?:\W+\w+){0,3}?\W+"
          r"(?:ignorier\w*|vergessen|irrelevant|hinter sich)\b",
          "German: the previous instructions are irrelevant"),
    _rule("OVR-DE-3", "instruction_override", 0.45,
          r"\bneue (?:anweisung\w*|aufgabe\w*)\b[^.\n]{0,20}(?:folg\w*|:)|\bnun folgen (?:neue|weitere)\b",
          "German: new instructions follow"),
    _rule("OVR-IT-1", "instruction_override", 0.9,
          r"\b(?:ignora|ignorate|ignori|dimentica|dimenticate|trascura|scorda)\w*\b(?:\W+\w+){0,4}?\W+"
          r"(?:istruzion\w*|indicazion\w*|regole|comandi|ordini|tutto|compiti)\b",
          "Italian: ignore or forget the instructions"),
    _rule("OVR-ES-FR", "instruction_override", 0.85,
          r"\b(?:olvida|olvide|olvidad|ignora|oubliez|oublie|ignorez)\b(?:\W+\w+){0,4}?\W+"
          r"(?:instrucciones|instructions|consignes|todo|tout)\b",
          "Spanish or French: forget all the instructions"),
    # --- role hijack ---------------------------------------------------------------------------------------
    _rule("ROLE-1", "role_hijack", 0.6, r"\byou are (?:now|no longer|from now on|henceforth)\b", "'You are now ...'"),
    _rule("ROLE-2", "role_hijack", 0.5, r"\bfrom now on,? (?:you|your)\b", "'From now on you ...'"),
    _rule("ROLE-3", "role_hijack", 0.8,
          r"\b(?:developer|god|admin|sudo|jailbreak|dan|unrestricted|unfiltered|evil) mode\b|\bdo anything now\b|"
          r"\bjailbr(?:eak|oken)\b",
          "Known jailbreak personas"),
    _rule("ROLE-4", "role_hijack", 0.5, r"\b(?:pretend|imagine)(?: that)? (?:you are|you're|to be)\b",
          "'Pretend you are ...'"),
    _rule("ROLE-5", "role_hijack", 0.5, r"\bi want you to (?:act|behave|respond|pretend|play|be)\b",
          "'I want you to act as ...'"),
    _rule("ROLE-6", "role_hijack", 0.6,
          r"\b(?:act|behave|respond|function) as (?:an? |the )?(?:linux terminal|python interpreter|interpreter|"
          r"terminal|unrestricted|unfiltered|evil|different|new)\b",
          "Re-purposes the model"),
    _rule("ROLE-7", "role_hijack", 0.6,
          r"\b(?:no|without) (?:programming |content |ethical |safety )?"
          r"(?:guidelines|restrictions|filters|limitations)\b",
          "Claims the model has no restrictions"),
    _rule("ROLE-DE", "role_hijack", 0.55,
          r"\b(?:du bist|sie sind) (?:jetzt|nun|ab sofort|ab jetzt)\b|\b(?:jetzt|nun|ab sofort) bist du\b|"
          r"\bstell dir vor,? du bist\b|\btu so,? als\b|\bdass (?:sie|du) als\b[^.\n]{0,60}\bfungier\w*",
          "German role change"),
    _rule("ROLE-IT", "role_hijack", 0.55,
          r"\b(?:ora|adesso|da ora in poi|d'ora in poi) (?:sei|tu sei|diventi)\b|\bfingi di essere\b",
          "Italian role change"),
    # --- system prompt exfiltration ------------------------------------------------------------------------
    _rule("EXF-1", "prompt_exfiltration", 0.85,
          r"\b(?:reveal|show|print|repeat|output|display|tell me|give me|share|leak|disclose|write out|spell out|list|"
          r"dump|copy)\b(?:\W+\w+){0,5}?\W+(?:system prompt|system message|initial prompt|initial instructions|"
          r"original prompt|original instructions|hidden prompt|hidden instructions|prompt texts?|full prompt|"
          r"your (?:instructions|prompt|rules|guidelines|configuration)|the prompt (?:above|text))\b",
          "Asks for the system prompt"),
    _rule("EXF-2", "prompt_exfiltration", 0.7,
          r"\bwhat (?:is|are|was|were) (?:your|the) (?:initial |original |system |exact |first )?"
          r"(?:instructions|prompt|rules)\b|\bwhat (?:was|is) written (?:above|at the beginning|before)\b",
          "Asks what the instructions were"),
    _rule("EXF-DE", "prompt_exfiltration", 0.8,
          r"\b(?:zeig\w*|gib|nenn\w*|wiederhol\w*|verrat\w*)\b[^.\n]{0,40}(?:system-?prompt|prompt-?text|"
          r"deine anweisungen|ihre anweisungen)",
          "German: show the prompt"),
    _rule("EXF-IT", "prompt_exfiltration", 0.8,
          r"\b(?:mostra|rivela|ripeti|stampa|dimmi)\w*\b[^.\n]{0,40}(?:prompt di sistema|istruzioni (?:iniziali|"
          r"originali|di sistema)|le tue istruzioni)",
          "Italian: reveal the prompt"),
    # --- tool-call coercion --------------------------------------------------------------------------------
    _rule("TOOL-1", "tool_coercion", 0.4,
          r"\b(?:call|invoke|execute|run|trigger)\b(?:\W+\w+){0,3}?\W+(?:tool|function|plugin|api|endpoint)s?\b",
          "Tells the model to call a tool"),
    _rule("TOOL-2", "tool_coercion", 0.45,
          r"\b(?:send|forward|e-?mail|mail|post|upload|transmit|exfiltrate|leak)\b[^.\n]{0,80}?\b(?:to|at|into)\b\s*"
          r"(?:\S+@\S+|<EMAIL_[0-9a-f]+>|https?://\S+)",
          "Sends data to an address or URL"),
    _rule("TOOL-3", "tool_coercion", 0.6,
          r"\{\s*\"(?:name|function|tool|tool_name)\"\s*:\s*\"[\w.-]+\"\s*,\s*\"(?:arguments|parameters|args|input)\"\s*:",
          "Embedded function-call JSON"),
    _rule("TOOL-4", "tool_coercion", 0.35,
          r"\b(?:transfer|wire|pay|refund)\b[^.\n]{0,20}?(?:[$€£]\s?\d|\d[\d.,]*\s?(?:eur|euro|usd|dollars|gbp)\b)",
          "Moves money"),
    _rule("TOOL-5", "tool_coercion", 0.5,
          r"\b(?:delete|drop|erase|wipe|truncate)\b[^.\n]{0,20}\b(?:database|table|all (?:files|records|data)|"
          r"repository|repo)\b",
          "Destructive operation"),
    _rule("TOOL-6", "tool_coercion", 0.45, r"\bcurl\s+(?:-\S+\s+)*https?://|\bwget\s+https?://",
          "Shell download command"),
    # --- conversation spoofing ------------------------------------------------------------------------------
    _rule("SPOOF-1", "conversation_spoofing", 0.8,
          r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>|\[/?INST\]|<</?SYS>>|</?(?:system|instructions?|"
          r"admin)>|###\s*(?:system|instruction|new instructions)\b",
          "Chat-template or role markers"),
    _rule("SPOOF-2", "conversation_spoofing", 0.45, r"(?:^|\n)\s*(?:system|assistant|developer|admin)\s*:\s*\S",
          "Fake conversation turn"),
    _rule("SPOOF-3", "conversation_spoofing", 0.4,
          r"\bend of (?:document|context|input|user input|text|data|instructions)\b", "Fake end-of-input marker"),
    _rule("SPOOF-4", "conversation_spoofing", 0.9, r"«\s*/?\s*untrusted", "Forged spotlighting delimiter"),
    # --- secrecy -------------------------------------------------------------------------------------------
    _rule("SECRECY-1", "secrecy", 0.45,
          r"\b(?:do not|don'?t|never|without)\b[^.\n]{0,20}\b(?:tell|telling|inform|informing|mention|mentioning|"
          r"notify|notifying|alert|alerting)\b[^.\n]{0,20}\b(?:the user|users?|anyone|anybody|the human|"
          r"the operator|the customer)\b",
          "Asks to hide the action from the user"),
)


@dataclass(frozen=True)
class RuleMatch:
    """A rule that matched, with a short excerpt around the match."""

    rule: Rule
    excerpt: str


def match_rules(text: str, rules: tuple[Rule, ...] = RULES, *, excerpt_chars: int = 80) -> list[RuleMatch]:
    """All rules that match ``text`` (first match per rule)."""
    matches: list[RuleMatch] = []
    for rule in rules:
        m = rule.pattern.search(text)
        if m:
            pad = max(0, (excerpt_chars - (m.end() - m.start())) // 2)
            excerpt = text[max(0, m.start() - pad) : m.end() + pad].replace("\n", " ").strip()
            matches.append(RuleMatch(rule, excerpt[: excerpt_chars + 20]))
    return matches


def combine(weights: list[float]) -> float:
    """Noisy-OR combination of independent signals."""
    remaining = 1.0
    for w in weights:
        remaining *= 1.0 - w
    return 1.0 - remaining


# ------------------------------------------------------------------------------------- link exfiltration

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?(https?://[^\s)>]+)")
_MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?(https?://[^\s)>]+)")
_HTML_IMG = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']?(https?://[^\s\"'>]+)", re.I)
_MD_REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*<?(https?://[^\s>]+)", re.M)


def host_allowed(url: str, allowlist: tuple[str, ...]) -> bool:
    """True when the URL's host equals an allowlisted domain or is a subdomain of one."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    for entry in allowlist:
        domain = entry.lower().lstrip("*").lstrip(".")
        if host == domain or host.endswith("." + domain):
            return True
    return False


@dataclass(frozen=True)
class LinkFinding:
    """A link or image pointing outside the allowlist."""

    url: str
    kind: str
    weight: float


def find_exfil_links(text: str, allowlist: tuple[str, ...]) -> list[LinkFinding]:
    """Images to non-allowlisted hosts load automatically when rendered, so they can leak data with zero clicks;
    links are reported only when they carry a query string long enough to hold data."""
    findings: list[LinkFinding] = []
    for pattern, kind in ((_MD_IMAGE, "image"), (_HTML_IMG, "image"), (_MD_LINK, "link"), (_MD_REFERENCE, "link")):
        for m in pattern.finditer(text):
            url = m.group(1)
            if host_allowed(url, allowlist):
                continue
            parts = urlsplit(url)
            carries_data = len(parts.query) >= 16 or len(parts.path) > 40
            if kind == "image":
                findings.append(LinkFinding(url, kind, 0.9 if carries_data or parts.query else 0.7))
            elif carries_data:
                findings.append(LinkFinding(url, kind, 0.5))
    return findings
