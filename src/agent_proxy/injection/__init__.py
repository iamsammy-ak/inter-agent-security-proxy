"""Prompt-injection defence for content passed between agents."""

from .classifier import InjectionClassifier, load_classifier
from .detector import InjectionDetector, InjectionVerdict, RuleHit
from .spotlight import Spotlighted, Spotlighter, system_notice
from .unicode import HiddenText, reveal_hidden

__all__ = [
    "HiddenText",
    "InjectionClassifier",
    "InjectionDetector",
    "InjectionVerdict",
    "RuleHit",
    "Spotlighted",
    "Spotlighter",
    "load_classifier",
    "reveal_hidden",
    "system_notice",
]
