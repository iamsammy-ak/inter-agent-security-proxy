"""Combine rules, hidden-text analysis, decoded payloads, link checks and the classifier into one verdict."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .classifier import InjectionClassifier
from .decode import decode_payloads
from .rules import RULES, Rule, combine, find_exfil_links, match_rules
from .unicode import normalise_for_matching, reveal_hidden


class RuleHit(BaseModel):
    """One signal that contributed to the verdict. ``view`` says where it was found (``text``, ``tag-chars``,
    ``base64`` ...)."""

    rule_id: str
    category: str
    weight: float
    view: str = "text"
    excerpt: str = ""


class InjectionVerdict(BaseModel):
    """Outcome of :meth:`InjectionDetector.inspect`."""

    is_injection: bool
    score: float = Field(description="max(rule score, classifier probability)")
    rule_score: float
    classifier_score: float | None = None
    classifier_threshold: float | None = None
    hits: list[RuleHit] = Field(default_factory=list)
    flagged_urls: list[str] = Field(default_factory=list)

    def categories(self) -> list[str]:
        return sorted({h.category for h in self.hits})


class InjectionDetector:
    """Rule set plus optional classifier. Either component can be switched off for ablation benchmarks."""

    def __init__(
        self,
        classifier: InjectionClassifier | None = None,
        *,
        use_rules: bool = True,
        rule_threshold: float = 0.5,
        link_allowlist: tuple[str, ...] = (),
        rules: tuple[Rule, ...] = RULES,
    ) -> None:
        self.classifier = classifier
        self.use_rules = use_rules
        self.rule_threshold = rule_threshold
        self.link_allowlist = link_allowlist
        self.rules = rules

    def inspect(self, text: str) -> InjectionVerdict:
        hidden = reveal_hidden(text)
        hits: list[RuleHit] = []
        flagged_urls: list[str] = []
        views: list[str] = [hidden.cleaned]

        if self.use_rules:
            for match in match_rules(normalise_for_matching(hidden.cleaned), self.rules):
                hits.append(RuleHit(rule_id=match.rule.id, category=match.rule.category, weight=match.rule.weight,
                                    excerpt=match.excerpt))
            # Hidden carriers: anything smuggled in invisible characters is suspicious on its own and is re-scanned.
            for view, payload in (("tag-chars", hidden.tag_payload), ("variation-selectors", hidden.selector_payload)):
                if payload:
                    views.append(payload)
                    inner = match_rules(normalise_for_matching(payload), self.rules)
                    hits.append(RuleHit(rule_id="HIDDEN-PAYLOAD", category="hidden_unicode",
                                        weight=0.9 if inner else 0.7, view=view, excerpt=payload[:80]))
                    hits.extend(RuleHit(rule_id=m.rule.id, category=m.rule.category, weight=m.rule.weight, view=view,
                                        excerpt=m.excerpt) for m in inner)
            if hidden.zero_width:
                hits.append(RuleHit(rule_id="HIDDEN-ZW", category="hidden_unicode", weight=0.35,
                                    excerpt=f"{hidden.zero_width} zero-width character(s)"))
            if hidden.bidi_controls:
                hits.append(RuleHit(rule_id="HIDDEN-BIDI", category="hidden_unicode", weight=0.6,
                                    excerpt=f"{hidden.bidi_controls} bidirectional control(s)"))
            for decoded in decode_payloads(hidden.cleaned):
                views.append(decoded.text)
                inner = match_rules(normalise_for_matching(decoded.text), self.rules)
                if inner:
                    hits.append(RuleHit(rule_id="ENCODED-INSTRUCTION", category="encoded_payload", weight=0.9,
                                        view=decoded.encoding, excerpt=decoded.text[:80]))
                    hits.extend(RuleHit(rule_id=m.rule.id, category=m.rule.category, weight=m.rule.weight,
                                        view=decoded.encoding, excerpt=m.excerpt) for m in inner)
                else:
                    hits.append(RuleHit(rule_id="ENCODED-TEXT", category="encoded_payload", weight=0.3,
                                        view=decoded.encoding, excerpt=decoded.text[:80]))
            for link in find_exfil_links(hidden.cleaned, self.link_allowlist):
                flagged_urls.append(link.url)
                hits.append(RuleHit(rule_id=f"EXFIL-{link.kind.upper()}", category="markdown_exfiltration",
                                    weight=link.weight, excerpt=link.url[:80]))

        rule_score = combine([h.weight for h in hits]) if hits else 0.0
        classifier_score = self.classifier.score(views) if self.classifier else None
        threshold = self.classifier.threshold if self.classifier else None
        flagged = (self.use_rules and rule_score >= self.rule_threshold) or (
            classifier_score is not None and threshold is not None and classifier_score >= threshold
        )
        return InjectionVerdict(
            is_injection=flagged,
            score=round(max(rule_score, classifier_score or 0.0), 4),
            rule_score=round(rule_score, 4),
            classifier_score=None if classifier_score is None else round(classifier_score, 4),
            classifier_threshold=threshold,
            hits=hits,
            flagged_urls=flagged_urls,
        )
