"""
Configuration for the PDF Extractor backend.
"""
import os

# -- Ollama Models --------------------------------------------------
TEXT_MODEL = "gemma3:4b"         
EMBEDDING_MODEL = "embeddinggemma" # Embedding model

# -- Reranker (optional) -------------------------------------------
ENABLE_RERANKING = True
RERANKER_MODEL = "dengcao/Qwen3-Reranker-0.6B:Q8_0"
RERANK_CANDIDATES = 20
RERANK_BATCH_SIZE = 8

# -- Extraction Policy ----------------------------------------------
# If True: accuracy-first (strict). If False: permissive (do not enforce enum allowed-values).
ACCURACY_FIRST = True

MIN_EXTRACT_CONFIDENCE = 0.5

# -- Chunking ------------------------------------------------------
CHUNK_SIZE = 400          # tokens (~375 words)
CHUNK_OVERLAP = 80       # 20% overlap
CHARS_PER_TOKEN = 4       # approximate characters per token

# -- ChromaDB ------------------------------------------------------
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "pdf_chunks"

# -- PDF Processing -------------------------------------------------
IMAGE_DPI = 300           # DPI for rendering PDF pages as images
TOP_K_CHUNKS = 5         # Number of chunks to retrieve per field


# -- Server ---------------------------------------------------------
UPLOAD_DIR = "./uploads"
LLM_MAX_WORKERS = 2       # Number of simultaneous fields to extract (safer for 8GB GPU)

# -- Magic Numbers / Extract Config --------------------------------
TABLE_CHUNK_OFFSET = 1000
ENABLE_REGEX_EXTRACTION = False

# -- Centralized Logging --------------------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pdf_extractor")
