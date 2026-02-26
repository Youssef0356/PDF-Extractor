"""
Embedding service using Ollama's nomic-embed-text model.
"""
import ollama as ollama_client

from config import EMBEDDING_MODEL


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for a text string.
    
    Args:
        text: The text to embed.
        
    Returns:
        List of floats representing the embedding vector.
    """
    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response["embeddings"][0]


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a batch.
    
    Args:
        texts: List of text strings.
        
    Returns:
        List of embedding vectors.
    """
    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return response["embeddings"]
