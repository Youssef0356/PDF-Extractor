"""
Blended confidence scoring for LLM field extractions.

The LLM (local Ollama model) is NOT calibrated — it almost always returns
confidence=0.9 or 1.0 regardless of actual certainty. This module replaces
that raw score with a multi-signal blend that produces meaningful 0–1 values:

Signals used (all weighted):
  1. chunk_distance  — ChromaDB cosine distance of best retrieved chunk (0=perfect)
  2. quote_verified  — Does the LLM's quoted text actually appear in the chunks?
  3. value_quality   — Is the value "Autre" / null / suspiciously short?
  4. llm_raw         — LLM self-reported confidence (used lightly, it's unreliable)
  5. rerank_score    — Cross-encoder reranker score when available

Output ranges:
  ≥ 0.90  → green  (confident — shown as 90%+)
  0.60–0.89 → amber (uncertain — shown as 60–89%)
  < 0.60  → red    (low confidence — shown below 60%)
"""

from __future__ import annotations
import math
from typing import Any


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
W_DISTANCE   = 0.35   # best chunk cosine distance
W_QUOTE      = 0.25   # quote found in retrieved context
W_VALUE      = 0.20   # value quality (not Autre, not null, not garbage)
W_LLM        = 0.10   # raw LLM self-reported confidence
W_RERANK     = 0.10   # cross-encoder rerank score when available


def compute_confidence(
    field_name: str,
    value: Any,
    llm_raw_confidence: float,
    quote: str,
    chunks: list[dict],
    *,
    is_enum: bool = False,
) -> float:
    """
    Compute a blended confidence score for one extracted field.

    Args:
        field_name:           Name of the field being extracted.
        value:                The normalized extracted value.
        llm_raw_confidence:   Confidence returned by the LLM (0–1).
        quote:                Verbatim text the LLM cited as evidence.
        chunks:               List of retrieved chunks (each has 'text',
                              'distance', optionally 'rerank_score').
        is_enum:              True if the field has a fixed set of allowed values.

    Returns:
        Float in [0, 1].
    """

    # ── 1. Distance score ──────────────────────────────────────────────────
    # ChromaDB cosine distance: 0 = identical, 2 = opposite.
    # Typical good retrieval is 0.20–0.45; > 0.70 is very weak.
    best_distance = min((c.get("distance", 1.0) for c in chunks), default=1.0)
    # Map to [0,1]: distance 0.0 → 1.0 score, distance 0.7 → 0.0 score
    distance_score = max(0.0, 1.0 - (best_distance / 0.70))

    # ── 2. Quote verification score ────────────────────────────────────────
    quote_score = 0.0
    if quote and isinstance(quote, str) and len(quote.strip()) >= 2:
        q_lower = quote.strip().lower()
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            if q_lower in chunk_text:
                quote_score = 1.0
                break
            # Partial match — at least half the quote words found
            q_words = [w for w in q_lower.split() if len(w) > 2]
            if q_words:
                found = sum(1 for w in q_words if w in chunk_text)
                partial = found / len(q_words)
                if partial > quote_score:
                    quote_score = partial * 0.75   # partial match capped at 0.75

    # ── 3. Value quality score ─────────────────────────────────────────────
    value_score = _value_quality(field_name, value, is_enum)

    # ── 4. LLM raw confidence (lightly trusted) ────────────────────────────
    # Squash it toward 0.5 to reduce over-confidence: f(x) = 0.5 + (x-0.5)*0.4
    llm_score = 0.5 + (float(llm_raw_confidence or 0.5) - 0.5) * 0.4

    # ── 5. Reranker score ──────────────────────────────────────────────────
    # Cross-encoder scores are log-odds; typical good score ≈ 5–10, bad ≈ -5.
    best_rerank = max(
        (c.get("rerank_score", None) for c in chunks if c.get("rerank_score") is not None),
        default=None,
    )
    if best_rerank is not None:
        # Sigmoid to map to [0,1]: σ(x/3) gives 0.95 at x=8, 0.5 at x=0
        rerank_score = 1.0 / (1.0 + math.exp(-best_rerank / 3.0))
    else:
        # No reranker available — redistribute its weight to distance
        rerank_score = distance_score   # use distance as proxy

    # ── Blend ──────────────────────────────────────────────────────────────
    blended = (
        W_DISTANCE * distance_score +
        W_QUOTE    * quote_score    +
        W_VALUE    * value_score    +
        W_LLM      * llm_score      +
        W_RERANK   * rerank_score
    )

    # ── Special overrides ──────────────────────────────────────────────────
    # If value is None or "Autre" with no suffix → cap at 0.45
    if value is None:
        return 0.0
    if isinstance(value, str) and value.strip() == "Autre":
        blended = min(blended, 0.45)
    # If no chunks at all (ISA tag rule) → full confidence already set upstream
    if not chunks:
        return min(1.0, max(0.0, blended))

    return round(min(1.0, max(0.0, blended)), 4)


def _value_quality(field_name: str, value: Any, is_enum: bool) -> float:
    """Score 0–1 based on how believable the extracted value looks."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 0.85   # booleans are easy to get right
    if isinstance(value, (int, float)):
        return 0.80   # numbers that survived normalization are usually correct

    s = str(value).strip()
    if not s:
        return 0.0

    # Bare "Autre" with no suffix = LLM gave up
    if s == "Autre":
        return 0.20
    # "Autre: X" = LLM found something but it wasn't in the enum
    if s.startswith("Autre:"):
        return 0.45

    # For enum fields, a clean match is high confidence
    if is_enum:
        return 0.90

    # Open text: penalize very short or suspiciously long values
    length = len(s)
    if length < 2:
        return 0.20
    if length > 200:
        return 0.50   # probably a hallucinated paragraph

    return 0.80
