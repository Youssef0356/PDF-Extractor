"""
Embedding service using Ollama's nomic-embed-text model.
"""
from __future__ import annotations

import ollama as ollama_client

from config import EMBEDDING_MODEL


_EMBEDDING_DIM: int | None = None


def _get_embedding_dim() -> int:
    global _EMBEDDING_DIM
    if _EMBEDDING_DIM is not None:
        return _EMBEDDING_DIM
    try:
        resp = ollama_client.embed(model=EMBEDDING_MODEL, input="dim_probe")
        emb = (resp.get("embeddings") or [[]])[0]
        if isinstance(emb, list) and len(emb) > 0:
            _EMBEDDING_DIM = len(emb)
            return _EMBEDDING_DIM
    except Exception:
        pass
    # Conservative fallback: nomic-embed-text is commonly 768.
    _EMBEDDING_DIM = 768
    return _EMBEDDING_DIM


def _zero_embedding() -> list[float]:
    return [0.0] * _get_embedding_dim()


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i in range(dim):
            out[i] += float(v[i])
    n = float(len(vectors))
    if n == 0:
        return out
    return [x / n for x in out]


def _split_long_text(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Split long text into smaller parts to avoid exceeding embedding context limits.

    Uses a simple char window with overlap; this is only for embedding safety.
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    parts: list[str] = []
    start = 0
    while start < len(t):
        end = min(start + max_chars, len(t))
        parts.append(t[start:end])
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return parts


def _embed_text_safe(text: str) -> list[float]:
    """Embed a single text, automatically splitting if needed."""
    parts = _split_long_text(text)
    if not parts:
        return _zero_embedding()
    try:
        response = ollama_client.embed(model=EMBEDDING_MODEL, input=parts)
        vectors = response.get("embeddings") or []
        if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
            m = _mean_vector([v for v in vectors if isinstance(v, list) and len(v) > 0])
            if m:
                return m
    except Exception:
        pass

    # Fallback: embed parts one-by-one (slower, but robust)
    vectors: list[list[float]] = []
    for p in parts:
        try:
            resp = ollama_client.embed(model=EMBEDDING_MODEL, input=p)
            emb = (resp.get("embeddings") or [[]])[0]
            if emb:
                vectors.append(emb)
        except Exception:
            continue
    m = _mean_vector(vectors)
    return m if m else _zero_embedding()


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for a text string.
    
    Args:
        text: The text to embed.
        
    Returns:
        List of floats representing the embedding vector.
    """
    return _embed_text_safe(text)


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a batch.
    
    Args:
        texts: List of text strings.
        
    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    # First try to embed in one batch for performance.
    try:
        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(texts):
            if all(isinstance(e, list) and len(e) > 0 for e in embeddings):
                return embeddings
    except Exception:
        pass

    # Fallback: per-item safe embedding (handles overlong inputs)
    return [_embed_text_safe(t) for t in texts]
