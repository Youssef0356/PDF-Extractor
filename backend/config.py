"""
Configuration for the PDF Extractor backend.
"""

# -- Ollama Models --------------------------------------------------
VISION_MODEL = "qwen2.5vl:7b"       # Multimodal (vision + text)
TEXT_MODEL = "qwen2.5vl:7b"         # Using VL model for text too (handles both)
EMBEDDING_MODEL = "nomic-embed-text" # Embedding model

# -- Chunking ------------------------------------------------------
CHUNK_SIZE = 500          # tokens (~375 words)
CHUNK_OVERLAP = 100       # 20% overlap
CHARS_PER_TOKEN = 4       # approximate characters per token

# -- ChromaDB ------------------------------------------------------
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "pdf_chunks"

# -- PDF Processing -------------------------------------------------
IMAGE_DPI = 300           # DPI for rendering PDF pages as images
TOP_K_CHUNKS = 5          # Number of chunks to retrieve per field

# -- Server ---------------------------------------------------------
UPLOAD_DIR = "./uploads"
