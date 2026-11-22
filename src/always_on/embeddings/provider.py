"""Local zero-token embedding providers and vector math utilities."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D float vectors in [-1.0, 1.0]."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance in [0.0, 1.0] (clamped from 1 - max(0, sim))."""
    sim = max(0.0, cosine_similarity(a, b))
    return max(0.0, min(1.0, 1.0 - sim))


def vector_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize float32 1-D numpy vector to raw bytes for SQLite BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def bytes_to_vector(blob: bytes) -> np.ndarray:
    """Deserialize float32 1-D numpy vector from SQLite BLOB."""
    return np.frombuffer(blob, dtype=np.float32).copy()


class EmbeddingProvider(ABC):
    """Abstract interface for fast local zero-token embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of output vectors."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed a batch of strings into normalized float32 numpy vectors."""

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a normalized float32 numpy vector."""
        return self.embed_batch([text])[0]


class FastEmbedProvider(EmbeddingProvider):
    """ONNX Runtime CPU embedding provider powered by FastEmbed (zero PyTorch)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        sample = next(self._model.embed(["dimension probe"]))
        self._dim = int(len(sample))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        cleaned = [t if t.strip() else " " for t in texts]
        vectors: list[np.ndarray] = []
        for vec in self._model.embed(cleaned):
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
            vectors.append(arr)
        return vectors


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic lightweight bag-of-tokens + char-ngram embedding provider.

    Ideal for offline unit tests and zero-download dry runs. Semantically overlapping
    texts yield high cosine similarity; disjoint texts yield low cosine similarity.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = self._TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec

        features: list[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"w:{tok}", 2.0))
            if len(tok) >= 4:
                for i in range(len(tok) - 2):
                    features.append((f"c:{tok[i:i+3]}", 0.5))
        for i in range(len(tokens) - 1):
            features.append((f"b:{tokens[i]}_{tokens[i+1]}", 1.5))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(t) for t in texts]


def create_embedding_provider(provider: str, model_name: str) -> EmbeddingProvider:
    """Factory that instantiates the configured local embedding provider."""
    if provider == "mock":
        return HashEmbeddingProvider()
    if provider == "fastembed":
        return FastEmbedProvider(model_name=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider!r}")
