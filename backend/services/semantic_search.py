"""
Semantic search service.
Queries ChromaDB with field-specific queries to find relevant document chunks.
"""
from config import TOP_K_CHUNKS
from models.schema import FIELD_DESCRIPTIONS
from services.vector_store import query_chunks


_IDENTITY_FIELDS = {"marque", "modele", "reference", "equipmentName"}

# Accuracy-first: for some fields, narrative text is much less noisy than tables.
# We still allow fallback to mixed retrieval if we don't get enough hits.
_TEXT_FIRST_FIELDS = {
    *_IDENTITY_FIELDS,
    "technologie",
    "plageMesure",
    "alimentation",
    "nbFils",
}


def search_for_field(
    field_name: str,
    n_results: int = TOP_K_CHUNKS,
    doc_id: str | None = None,
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

    # Text-first fields: prefer narrative text first, then allow tables as fallback.
    if field_name in _TEXT_FIRST_FIELDS:
        where_text = dict(base_where or {})
        where_text["chunk_type"] = "text"
        for query in field_info["search_queries"]:
            _merge_results(query_chunks(query, n_results=n_results, where=where_text))

        if len(all_results) < max(1, n_results // 2):
            for query in field_info["search_queries"]:
                _merge_results(query_chunks(query, n_results=n_results, where=base_where))
    else:
        for query in field_info["search_queries"]:
            _merge_results(query_chunks(query, n_results=n_results, where=base_where))
    
    # Sort by distance (lower = more relevant)
    sorted_results = sorted(all_results.values(), key=lambda r: r["distance"])
    
    return sorted_results[:n_results]


def search_all_fields(doc_id: str | None = None) -> dict[str, list[dict]]:
    """
    Search for relevant chunks for ALL form fields.
    
    Returns:
        Dict mapping field_name -> list of relevant chunks.
    """
    results = {}
    for field_name in FIELD_DESCRIPTIONS:
        results[field_name] = search_for_field(field_name, doc_id=doc_id)
    return results
