"""
Configuration for the PDF Extractor backend.
"""
import os

# -- Ollama Models --------------------------------------------------
TEXT_MODEL = "gemma3:4b"         
EMBEDDING_MODEL = "nomic-embed-text"  # 335M multilingual embedding model

# -- Reranker (optional) -------------------------------------------
ENABLE_RERANKING = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20
RERANK_BATCH_SIZE = 8

# -- Extraction Policy ----------------------------------------------
# If True: accuracy-first (strict). If False: permissive (do not enforce enum allowed-values).
ACCURACY_FIRST = True

MIN_EXTRACT_CONFIDENCE = 0.7

# -- Chunking ------------------------------------------------------
CHUNK_SIZE = 500          # tokens (~375 words)
CHUNK_OVERLAP = 100       # 20% overlap
CHARS_PER_TOKEN = 4     # approximate characters per token (standard for most tokenizers)

# -- ChromaDB ------------------------------------------------------
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "pdf_chunks"

# -- PDF Processing -------------------------------------------------
IMAGE_DPI = 300           # DPI for rendering PDF pages as images
TOP_K_CHUNKS = 3         # Number of chunks to retrieve per field


# -- Server ---------------------------------------------------------
UPLOAD_DIR = "./uploads"
LLM_MAX_WORKERS = 1       # Number of simultaneous fields to extract (safer for 8GB GPU)

# -- Magic Numbers / Extract Config --------------------------------
TABLE_CHUNK_OFFSET = 1000
ENABLE_REGEX_EXTRACTION = True

# Re-retrieval verification (disabled by default - causes false negatives)
ENABLE_RERETRIEVAL_VERIFICATION = False

# -- Centralized Logging --------------------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pdf_extractor")
