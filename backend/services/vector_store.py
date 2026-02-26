"""
ChromaDB vector store service for storing and querying document chunks.
Supports streaming: insert small batches as they arrive rather than all at once.
"""
import chromadb

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from services.embeddings import generate_embedding, generate_embeddings_batch


# -- Initialize ChromaDB client -------------------------------------
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(name: str = CHROMA_COLLECTION_NAME):
    """Get or create a ChromaDB collection."""
    return _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection(name: str = CHROMA_COLLECTION_NAME):
    """Delete and recreate a collection (for re-processing a PDF)."""
    try:
        _client.delete_collection(name)
    except Exception:
        pass
    return get_or_create_collection(name)


def store_chunks_batch(chunks: list, collection) -> None:
    """
    Embed and store a small batch of chunks into an already-open collection.
    Call this repeatedly during streaming instead of store_chunks() once.

    Args:
        chunks: List of Chunk objects (can be just one page worth).
        collection: An open chromadb collection object.
    """
    if not chunks:
        return

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "has_images": chunk.has_images,
        }
        for chunk in chunks
    ]

    embeddings = generate_embeddings_batch(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def store_chunks(chunks: list, collection_name: str = CHROMA_COLLECTION_NAME):
    """
    Legacy: embed and store all chunks at once (used by main.py).
    For large PDFs, prefer streaming with store_chunks_batch().
    """
    collection = clear_collection(collection_name)
    if not chunks:
        return
    store_chunks_batch(chunks, collection)
    print(f"[VectorStore] Stored {len(chunks)} chunks in '{collection_name}'")


def query_chunks(
    query_text: str,
    n_results: int = 5,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[dict]:
    """
    Query ChromaDB for the most relevant chunks.

    Args:
        query_text: The search query.
        n_results: Number of results to return.
        collection_name: ChromaDB collection name.

    Returns:
        List of dicts with 'text', 'metadata', and 'distance'.
    """
    collection = get_or_create_collection(collection_name)

    query_embedding = generate_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    formatted = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            formatted.append({
                "text": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })

    return formatted
