"""
Configuration for the PDF Extractor backend.
"""
import os

# -- Ollama Models --------------------------------------------------
VISION_MODEL = "qwen2.5vl:3b"       # Multimodal (vision + text)
TEXT_MODEL = "gemma3:4b"         # Using VL model for text too (handles both)
EMBEDDING_MODEL = "nomic-embed-text" # Embedding model

# -- Extraction Policy ----------------------------------------------
# If True: accuracy-first (strict). If False: permissive (do not enforce enum allowed-values).
ACCURACY_FIRST = False

# -- Chunking ------------------------------------------------------
CHUNK_SIZE = 500          # tokens (~375 words)
CHUNK_OVERLAP = 100       # 20% overlap
CHARS_PER_TOKEN = 4       # approximate characters per token

# -- ChromaDB ------------------------------------------------------
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "pdf_chunks"

# -- PDF Processing -------------------------------------------------
IMAGE_DPI = 300           # DPI for rendering PDF pages as images
TOP_K_CHUNKS = 5          # Number of chunks to retrieve per field

ENABLE_VISION_OCR = False
VISION_OCR_ONLY_IF_NO_TEXT = False
VISION_OCR_MAX_PAGES = 50
VISION_OCR_TEXT_CHAR_THRESHOLD = 200
VISION_OCR_AUTO_MIXED_PAGES = False
VISION_OCR_TEXT_AREA_RATIO_MAX = 0.15
VISION_OCR_MAX_IMAGE_AREA_RATIO_MIN = 0.25
VISION_OCR_MAX_DRAWING_AREA_RATIO_MIN = 0.35
VISION_OCR_DRAWING_COUNT_MIN = 25

# -- Server ---------------------------------------------------------
UPLOAD_DIR = "./uploads"
LLM_MAX_WORKERS = 1       # Number of simultaneous fields to extract (safer for 8GB GPU)
