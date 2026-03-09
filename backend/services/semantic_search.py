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
import json
import ollama as ollama_client
from models.schema import FIELD_DESCRIPTIONS
from services.vector_store import query_chunks


_IDENTITY_FIELDS = {"marque", "modele", "reference", "equipmentName"}

# Accuracy-first: for some fields, narrative text is much less noisy than tables.
# We still allow fallback to mixed retrieval if we don't get enough hits.
_TEXT_FIRST_FIELDS = {
    *_IDENTITY_FIELDS,
    "technologie",
    "alimentation",
    "nbFils",
}


def search_for_field(
    field_name: str,
    n_results: int = TOP_K_CHUNKS,
    doc_id: str | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[dict]:
    """
    Search for chunks relevant to a specific form field.
    
    Uses the predefined search queries for each field to find
    the most semantically relevant document chunks.
    
    Args:
        field_name: Name of the field in the equipment schema.
        n_results: Number of chunks to retrieve.
        
    Returns:
        Deduplicated list of relevant chunks sorted by relevance.
    """
    field_info = FIELD_DESCRIPTIONS.get(field_name)
    if not field_info:
        return []
    
    # Run multiple queries and merge results
    all_results: dict[str, dict] = {}

    def _key_for(r: dict) -> str:
        md = (r.get("metadata") or {})
        cid = md.get("chunk_id")
        if isinstance(cid, str) and cid:
            return cid
        return r.get("text") or ""

    def _merge_results(results: list[dict]) -> None:
        for result in results:
            k = _key_for(result)
            if not k:
                continue
            if k not in all_results or result["distance"] < all_results[k]["distance"]:
                all_results[k] = result

    base_where = {"doc_id": doc_id} if doc_id else None

    def _rerank(query: str, results: list[dict], limit: int) -> list[dict]:
        if not ENABLE_RERANKING:
            return results[:limit]
        if not results or limit <= 0:
            return []

        texts: list[str] = []
        for r in results:
            t = (r.get("text") or "").strip()
            if t:
                texts.append(t)
            else:
                texts.append("")

        scored: list[tuple[float, int]] = []

        def _extract_first_json_array(text: str) -> str | None:
            t = (text or "").strip()
            if not t:
                return None
            if t.startswith("```"):
                t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("```"))
            start = t.find("[")
            if start < 0:
                return None
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(t)):
                ch = t[i]
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return t[start:i + 1]
            return None

        for start in range(0, len(texts), max(1, int(RERANK_BATCH_SIZE))):
            batch_texts = texts[start:start + max(1, int(RERANK_BATCH_SIZE))]
            messages = [
                {
                    "role": "user",
                    "content": "You are a reranker. Score each document for relevance to the query. Return ONLY a JSON array of numbers (floats) with same length as documents.\n"
                    f"Query: {query}\n"
                    "Documents:\n" + "\n".join([f"[{i}] {d}" for i, d in enumerate(batch_texts)]),
                }
            ]
            try:
                resp = ollama_client.chat(
                    model=RERANKER_MODEL,
                    messages=messages,
                    options={"temperature": 0.0},
                    keep_alive="10m",
                    format="json",
                )
                raw = (resp.get("message") or {}).get("content") or ""
                payload = _extract_first_json_array(raw) or raw
                scores = json.loads(payload)
                if not isinstance(scores, list):
                    raise ValueError("reranker did not return a list")
                for i, s in enumerate(scores):
                    try:
                        scored.append((float(s), start + i))
                    except Exception:
                        scored.append((0.0, start + i))
            except Exception:
                for i in range(len(batch_texts)):
                    scored.append((0.0, start + i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]
        reranked = [results[idx] for _, idx in top if 0 <= idx < len(results)]
        return reranked

    # Text-first fields: prefer narrative text first, then allow tables as fallback.
    if field_name in _TEXT_FIRST_FIELDS:
        # Include OCR chunks as narrative text too.
        for ct in ("text", "ocr"):
            conditions = [{"chunk_type": ct}]
            if doc_id:
                conditions.append({"doc_id": doc_id})
            
            where_text = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            
            for query in field_info["search_queries"]:
                _merge_results(
                    query_chunks(
                        query,
                        n_results=max(n_results, int(RERANK_CANDIDATES)) if ENABLE_RERANKING else n_results,
                        where=where_text,
                        collection_name=collection_name,
                    )
                )

        if len(all_results) < max(1, n_results // 2):
            for query in field_info["search_queries"]:
                _merge_results(
                    query_chunks(
                        query,
                        n_results=max(n_results, int(RERANK_CANDIDATES)) if ENABLE_RERANKING else n_results,
                        where=base_where,
                        collection_name=collection_name,
                    )
                )
    else:
        for query in field_info["search_queries"]:
            _merge_results(
                query_chunks(
                    query,
                    n_results=max(n_results, int(RERANK_CANDIDATES)) if ENABLE_RERANKING else n_results,
                    where=base_where,
                    collection_name=collection_name,
                )
            )
    
    # Sort by distance (lower = more relevant)
    sorted_results = sorted(all_results.values(), key=lambda r: r["distance"])

    if ENABLE_RERANKING and sorted_results:
        # Use the field's first search query as the rerank query anchor.
        rerank_query = (field_info.get("search_queries") or [field_name])[0]
        return _rerank(rerank_query, sorted_results, n_results)

    return sorted_results[:n_results]


def search_all_fields(
    doc_id: str | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    n_results: int = TOP_K_CHUNKS,
) -> dict[str, list[dict]]:
    """
    Search for relevant chunks for ALL form fields.
    
    Returns:
        Dict mapping field_name -> list of relevant chunks.
    """
    results = {}
    for field_name in FIELD_DESCRIPTIONS:
        results[field_name] = search_for_field(
            field_name,
            n_results=n_results,
            doc_id=doc_id,
            collection_name=collection_name,
        )
    return results
