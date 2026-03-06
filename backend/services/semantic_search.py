"""
Semantic search service.
Queries ChromaDB with field-specific queries to find relevant document chunks.
"""
from config import TOP_K_CHUNKS, CHROMA_COLLECTION_NAME
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
                        n_results=n_results,
                        where=where_text,
                        collection_name=collection_name,
                    )
                )

        if len(all_results) < max(1, n_results // 2):
            for query in field_info["search_queries"]:
                _merge_results(
                    query_chunks(
                        query,
                        n_results=n_results,
                        where=base_where,
                        collection_name=collection_name,
                    )
                )
    else:
        for query in field_info["search_queries"]:
            _merge_results(
                query_chunks(
                    query,
                    n_results=n_results,
                    where=base_where,
                    collection_name=collection_name,
                )
            )
    
    # Sort by distance (lower = more relevant)
    sorted_results = sorted(all_results.values(), key=lambda r: r["distance"])
    
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
