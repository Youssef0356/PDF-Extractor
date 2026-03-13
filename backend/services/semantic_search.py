import logging
import time
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_PERSIST_DIR as CHROMA_PATH,
    CHROMA_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL,
    ENABLE_RERANKING as USE_RERANKER,
    RERANKER_MODEL,
)
from models.schema_v2 import FIELD_DESCRIPTIONS_V2

logger = logging.getLogger(__name__)

# Initialize cross-encoder for reranking
reranker = None
if USE_RERANKER:
    try:
        reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Loaded reranker model: {RERANKER_MODEL}")
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")

_RERANK_INSTRUCTIONS = {
    "technologie": "Find the mention of the measurement technology (radar, ultrasonic, coriolis, electromagnetic, etc.)",
    "signalSortie": "Find the description of the output signal (4-20mA, HART, digital, relay, etc.)",
    "alimentation": "Find the power supply specification (24VDC, 230VAC, loop-powered, etc.)",
    "marque": "Find the name of the manufacturer or brand of the device",
    "plageMesureMax": "Find the upper limit, span, or maximum range of measurement",
}

# Fields where text search should be prioritized over vector search
_TEXT_FIRST_FIELDS = {"marque", "code", "technologie", "indiceIP"}

# Identity fields that help distinguish between similar products
_IDENTITY_FIELDS = {"marque", "category", "typeMesure", "code"}

class SemanticSearchService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        # nomic-embed-text is 768. 
        # The error "expecting 768, got 384" suggests Chroma's default embedding function (all-MiniLM-L6-v2) is being used.
        # We must NOT provide query_texts if we want to use our own embeddings, 
        # OR we must configure the collection with the correct embedding function.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def search_for_field(
        self, 
        field_name: str, 
        n_results: int = 5,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant document chunks for a specific field.
        """
        field_meta = FIELD_DESCRIPTIONS_V2.get(field_name)
        if not field_meta:
            return []

        queries = field_meta.get("search_queries", [field_name])
        
        # 1. Generate embeddings for queries manually to match the 768 dimension of nomic-embed-text
        from services.embeddings import generate_embeddings_batch
        query_embeddings = generate_embeddings_batch(queries)

        # 2. Try vector search using embeddings
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=metadata_filters
        )

        chunks = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                chunks.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else 1.0,
                    "id": results["ids"][0][i]
                })

        # 2. Rerank if enabled
        if reranker and chunks and field_name in _RERANK_INSTRUCTIONS:
            chunks = self._rerank(field_name, chunks)

        return chunks

    def _rerank(self, field_name: str, chunks: List[Dict]) -> List[Dict]:
        """Reranks chunks based on semantic relevance to the field's instruction."""
        if not chunks:
            return []
            
        instruction = _RERANK_INSTRUCTIONS.get(field_name, f"Find information about {field_name}")
        
        pairs = [[instruction, c["text"]] for c in chunks]
        scores = reranker.predict(pairs)
        
        for i, score in enumerate(scores):
            chunks[i]["rerank_score"] = float(score)
            
        # Re-sort
        return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def search_all_fields(self, metadata_filters: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """Returns a map of field_name -> relevant chunks for all extractable fields."""
        results = {}
        for field in FIELD_DESCRIPTIONS_V2.keys():
            if FIELD_DESCRIPTIONS_V2[field].get("ai_fills", True):
                results[field] = self.search_for_field(field, metadata_filters=metadata_filters)
        return results