"""
Semantic search service.
Queries ChromaDB with field-specific queries to find relevant document chunks.
"""
from config import TOP_K_CHUNKS
from models.schema import FIELD_DESCRIPTIONS
from services.vector_store import query_chunks


def search_for_field(field_name: str, n_results: int = TOP_K_CHUNKS) -> list[dict]:
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
    
    for query in field_info["search_queries"]:
        results = query_chunks(query, n_results=n_results)
        for result in results:
            chunk_text = result["text"]
            # Keep the best (lowest distance) result for each chunk
            if chunk_text not in all_results or result["distance"] < all_results[chunk_text]["distance"]:
                all_results[chunk_text] = result
    
    # Sort by distance (lower = more relevant)
    sorted_results = sorted(all_results.values(), key=lambda r: r["distance"])
    
    return sorted_results[:n_results]


def search_all_fields() -> dict[str, list[dict]]:
    """
    Search for relevant chunks for ALL form fields.
    
    Returns:
        Dict mapping field_name -> list of relevant chunks.
    """
    results = {}
    for field_name in FIELD_DESCRIPTIONS:
        results[field_name] = search_for_field(field_name)
    return results
