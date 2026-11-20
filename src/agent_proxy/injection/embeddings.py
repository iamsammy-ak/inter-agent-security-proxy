"""Text embedders for the injection classifier.

``FastEmbedEmbedder`` runs ``BAAI/bge-small-en-v1.5`` (384 dimensions, ONNX on CPU). ``HashingEmbedder`` needs no
download: signed feature hashing of word and character n-grams, L2-normalised. It is weaker, but keeps the
classifier usable in CI and on machines without the model.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class ModelUnavailable(RuntimeError):
    """The embedding model cannot be loaded (package missing or download impossible)."""


class Embedder(Protocol):
    """Maps texts to a 2-D float array of shape ``(len(texts), dim)``."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


_WORD = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Signed feature hashing of word unigrams, word bigrams and character 3-5-grams."""

    def __init__(self, dim: int = 2048, max_chars: int = 3000) -> None:
        self.dim = dim
        self.max_chars = max_chars
        self.name = f"hashing-v1:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        text = text[: self.max_chars].lower()
        words = _WORD.findall(text)
        features = [f"w:{w}" for w in words]
        features += [f"b:{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        padded = f" {' '.join(words)} "
        features += [f"c:{padded[i:i + n]}" for n in (3, 4, 5) for i in range(len(padded) - n + 1)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


class FastEmbedEmbedder:
    """ONNX sentence embeddings through ``fastembed``; honours ``FASTEMBED_CACHE_PATH``."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelUnavailable("fastembed is not installed (pip install '.[models]')") from exc
        try:
            self._model: Any = TextEmbedding(
                model_name=model_name, cache_dir=cache_dir or os.environ.get("FASTEMBED_CACHE_PATH")
            )
        except Exception as exc:  # pragma: no cover - download or ONNX failures
            raise ModelUnavailable(f"could not load {model_name}: {exc}") from exc
        self.name = f"fastembed:{model_name}"
        self.dim = 384

    def embed(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Embed in small batches of similar length: a batch is padded to its longest text, so mixing one long
        tool result with many short messages would make every row cost the full 512-token window."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = list(self._model.embed([texts[i] for i in order], batch_size=batch_size))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for position, index in enumerate(order):
            out[index] = vectors[position]
        return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    """Split long text into overlapping windows on whitespace, so an injection buried at the end of a long tool
    result still reaches the classifier (the embedding model truncates at 512 tokens)."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > 0 else end
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

