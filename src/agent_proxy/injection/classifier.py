"""Logistic-regression injection classifier over sentence embeddings.

Weights are trained by ``benchmarks/train_classifier.py`` on the deepset/prompt-injections *train* split and
shipped as small JSON artefacts, so inference needs only numpy (plus the embedding model).
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder, ModelUnavailable, chunk_text, sigmoid

log = logging.getLogger(__name__)

ClassifierName = Literal["bge-small", "hashing"]
ARTEFACT_FILES: dict[str, str] = {"bge-small": "classifier_bge-small.json", "hashing": "classifier_hashing.json"}


class ClassifierArtefact(BaseModel):
    """Serialised model: standardisation, coefficients, the decision threshold and how it was obtained."""

    name: str
    embedder: str
    dim: int
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    threshold: float
    C: float
    trained_on: str
    n_train: int
    cv_metrics: dict[str, float]


def artefact_path(name: str) -> Path:
    return Path(str(resources.files("agent_proxy") / "data" / ARTEFACT_FILES[name]))


def build_embedder(artefact: ClassifierArtefact) -> Embedder:
    if artefact.embedder.startswith("hashing-v1:"):
        return HashingEmbedder(dim=int(artefact.embedder.split(":", 1)[1]))
    if artefact.embedder.startswith("fastembed:"):
        return FastEmbedEmbedder(model_name=artefact.embedder.split(":", 1)[1])
    raise ValueError(f"unknown embedder {artefact.embedder!r}")


class InjectionClassifier:
    """Probability that a text is a prompt injection."""

    def __init__(self, artefact: ClassifierArtefact, embedder: Embedder) -> None:
        if embedder.dim != artefact.dim:
            raise ValueError(f"embedder dim {embedder.dim} does not match artefact dim {artefact.dim}")
        self.artefact = artefact
        self.embedder = embedder
        self._mean = np.asarray(artefact.mean, dtype=np.float32)
        self._scale = np.asarray(artefact.scale, dtype=np.float32)
        self._coef = np.asarray(artefact.coef, dtype=np.float32)

    @property
    def name(self) -> str:
        return self.artefact.name

    @property
    def threshold(self) -> float:
        return self.artefact.threshold

    @classmethod
    def load(cls, name: ClassifierName = "bge-small", *, path: Path | None = None) -> InjectionClassifier:
        """Load a shipped artefact (raises :class:`ModelUnavailable` if its embedding model cannot load)."""
        artefact = ClassifierArtefact.model_validate_json((path or artefact_path(name)).read_text(encoding="utf-8"))
        return cls(artefact, build_embedder(artefact))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability per text (texts longer than the model window are truncated by the embedder)."""
        features = (self.embedder.embed(texts) - self._mean) / self._scale
        return sigmoid(features @ self._coef + self.artefact.intercept)

    def score(self, texts: list[str]) -> float:
        """Highest probability over all texts and their overlapping chunks."""
        chunks = [chunk for text in texts if text.strip() for chunk in chunk_text(text)]
        if not chunks:
            return 0.0
        return float(np.max(self.predict_proba(chunks)))


def load_classifier(preference: str = "auto") -> InjectionClassifier | None:
    """``auto`` tries the embedding model and falls back to the hashing model; ``off`` disables the classifier."""
    if preference == "off":
        return None
    if preference in ("auto", "bge-small"):
        try:
            return InjectionClassifier.load("bge-small")
        except ModelUnavailable as exc:
            if preference == "bge-small":
                raise
            log.warning("embedding model unavailable (%s); using the hashing classifier", exc)
    return InjectionClassifier.load("hashing")
