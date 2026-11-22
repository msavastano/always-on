"""Local embedding providers and vector similarity helpers."""

from .provider import (
    EmbeddingProvider,
    FastEmbedProvider,
    HashEmbeddingProvider,
    bytes_to_vector,
    cosine_distance,
    cosine_similarity,
    create_embedding_provider,
    vector_to_bytes,
)

__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "HashEmbeddingProvider",
    "bytes_to_vector",
    "cosine_distance",
    "cosine_similarity",
    "create_embedding_provider",
    "vector_to_bytes",
]
