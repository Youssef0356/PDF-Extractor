"""
Semantic search service.
Queries ChromaDB with field-specific queries to find relevant document chunks.
"""
from config import (
    TOP_K_CHUNKS,
    CHROMA_COLLECTION_NAME,
    ENABLE_RERANKING,
    RERANKER_MODEL,
    RERANK_CANDIDATES,
    RERANK_BATCH_SIZE,
)
from typing import Any
from models.schema import FIELD_DESCRIPTIONS
from services.vector_store import query_chunks


_CROSS_ENCODER: Any | None = None


def _get_cross_encoder() -> Any:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL)
    return _CROSS_ENCODER


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS = {"marque", "modele", "reference", "equipmentName"}

# These fields appear more reliably in narrative text than in tables.
# We query text/ocr chunks first and fall back to all chunks if too few results.
_TEXT_FIRST_FIELDS = {
    *_IDENTITY_FIELDS,
    "technologie",
    "alimentation",
    "nbFils",
}

# Per-field reranking instructions (field-specific context improves score accuracy by 1-5%).
# Written in English as recommended by Qwen3 docs.
_RERANK_INSTRUCTIONS: dict[str, str] = {
    "equipmentName":    "Retrieve text that contains the commercial product name or device designation.",
    "marque":           "Retrieve text that mentions the manufacturer or brand name.",
    "modele":           "Retrieve text that contains the model name or model designation.",
    "reference":        "Retrieve text that contains an order number, part number, or article number.",
    "categorie":        "Retrieve text that describes what type of instrument or device this is.",
    "typeMesure":       "Retrieve text that states what physical quantity is being measured (flow, pressure, level, temperature).",
    "technologie":      "Retrieve text that explains the measurement principle or technology used.",
    "plageMesure":      "Retrieve text that states the measuring range, minimum and maximum values with units.",
    "typeSignal":       "Retrieve text that describes the output signal type (4-20mA, 0-10V, etc.).",
    "nbFils":           "Retrieve text that mentions 2-wire, 4-wire, or number of connection wires.",
    "alimentation":     "Retrieve text that states the power supply voltage or type.",
    "communication":    "Retrieve text that mentions a digital communication protocol (HART, Modbus, Profibus, etc.).",
    "indiceIP":         "Retrieve text that states the IP protection class or NEMA enclosure rating.",
    "reperage":         "Retrieve text that contains an instrument tag number or P&ID identifier.",
    "sortiesAlarme":    "Retrieve text that describes alarm outputs, relay contacts, or threshold setpoints.",
    "dateCalibration":  "Retrieve text that states the calibration interval, frequency, or schedule.",
}

_DEFAULT_RERANK_INSTRUCTION = "Retrieve the most relevant passage for the given query from an industrial equipment datasheet."


def _rerank(
    query: str,
    field_name: str,
    results: list[dict],
    limit: int,
) -> list[dict]:
    """
    Rerank retrieved chunks using a local CrossEncoder (sentence-transformers).

    Args:
        query:      The search query string used for retrieval.
        field_name: Used to select a field-specific instruction.
        results:    Candidate chunks from vector search (unsorted or sorted by distance).
        limit:      Number of top results to return after reranking.

    Returns:
        Top `limit` chunks reranked by relevance score (highest first).
    """
    if not ENABLE_RERANKING or not results or limit <= 0:
        return results[:limit]

    instruction = _RERANK_INSTRUCTIONS.get(field_name, _DEFAULT_RERANK_INSTRUCTION)
    print(f"  [RERANK] model={RERANKER_MODEL} field={field_name} candidates={len(results)} limit={limit}")

    texts: list[str] = [(r.get("text") or "").strip() for r in results]
    scored: list[tuple[float, int]] = []
    batch_size = max(1, int(RERANK_BATCH_SIZE))

    try:
        ce = _get_cross_encoder()
    except Exception as e:
        print(f"  [RERANK] failed to load CrossEncoder model={RERANKER_MODEL}: {e}")
        return results[:limit]

    query_with_instruction = f"{query}\n{instruction}".strip()

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        pairs = [(query_with_instruction, t) for t in batch_texts]

        try:
            scores = ce.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            scores_list = [float(s) for s in scores]
        except Exception as e:
            print(f"  [RERANK] failed batch start={batch_start} size={len(batch_texts)}: {e}")
            scores_list = [0.0] * len(batch_texts)

        for i, s in enumerate(scores_list):
            scored.append((float(s), batch_start + i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]
    reranked = [results[idx] for _, idx in top if 0 <= idx < len(results)]
    return reranked


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search_for_field(
    field_name: str,
    n_results: int = TOP_K_CHUNKS,
    doc_id: str | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[dict]:
    """
    Search for chunks relevant to a specific form field.

    Strategy:
    - Text-first fields: query text/ocr chunks first; fall back to all chunks if sparse.
    - Other fields: query all chunk types directly.
    - If reranking enabled: retrieve RERANK_CANDIDATES, rerank, return top n_results.

    Args:
        field_name:       Schema field name.
        n_results:        Final number of chunks to return.
        doc_id:           Filter to a specific document (recommended).
        collection_name:  ChromaDB collection name.

    Returns:
        Deduplicated list of chunks sorted by relevance (best first).
    """
    field_info = FIELD_DESCRIPTIONS.get(field_name)
    if not field_info:
        return []

    # How many candidates to fetch — more if reranking will filter them down
    fetch_n = max(n_results, int(RERANK_CANDIDATES)) if ENABLE_RERANKING else n_results

    base_where = {"doc_id": doc_id} if doc_id else None

    # Deduplicated result accumulator (keyed by chunk_id or text)
    all_results: dict[str, dict] = {}

    def _chunk_key(r: dict) -> str:
        md = r.get("metadata") or {}
        cid = md.get("chunk_id")
        if isinstance(cid, str) and cid:
            return cid
        return r.get("text") or ""

    def _merge(results: list[dict]) -> None:
        for r in results:
            k = _chunk_key(r)
            if not k:
                continue
            # Keep the result with the lower (better) vector distance
            if k not in all_results or r["distance"] < all_results[k]["distance"]:
                all_results[k] = r

    search_queries = field_info.get("search_queries", [field_name])

    if field_name in _TEXT_FIRST_FIELDS:
        # Pass 1: text and OCR chunks only
        for chunk_type in ("text", "ocr"):
            conditions = [{"chunk_type": chunk_type}]
            if doc_id:
                conditions.append({"doc_id": doc_id})
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            for query in search_queries:
                _merge(query_chunks(query, n_results=fetch_n, where=where, collection_name=collection_name))

        # Pass 2: fallback to all chunk types if we didn't get enough
        if len(all_results) < max(1, n_results // 2):
            for query in search_queries:
                _merge(query_chunks(query, n_results=fetch_n, where=base_where, collection_name=collection_name))
    else:
        for query in search_queries:
            _merge(query_chunks(query, n_results=fetch_n, where=base_where, collection_name=collection_name))

    # Sort by vector distance before reranking (lower = better)
    candidates = sorted(all_results.values(), key=lambda r: r["distance"])

    if ENABLE_RERANKING and candidates:
        # Use the most specific search query as the reranker anchor
        rerank_query = search_queries[0]
        return _rerank(rerank_query, field_name, candidates, n_results)

    return candidates[:n_results]


def search_all_fields(
    doc_id: str | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    n_results: int = TOP_K_CHUNKS,
) -> dict[str, list[dict]]:
    """
    Search for relevant chunks for ALL schema fields.

    Returns:
        Dict mapping field_name -> list of relevant chunks (best first).
    """
    return {
        field_name: search_for_field(
            field_name,
            n_results=n_results,
            doc_id=doc_id,
            collection_name=collection_name,
        )
        for field_name in FIELD_DESCRIPTIONS
    }