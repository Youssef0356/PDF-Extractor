# PDF Extractor

Extract structured equipment metadata from industrial PDF datasheets using a hybrid pipeline:

- PDF parsing (text + optional tables)
- Chunking
- Embeddings + vector search (ChromaDB)
- Optional deterministic regex extraction
- LLM extraction (Ollama)
- Evidence + per-field validation/meta

This repo contains:

- `backend/`: FastAPI API + CLI pipeline + extraction logic
- `frontend/`: React (Vite) UI (optional)
- `chroma_db/`: Local persistent ChromaDB storage (created/used at runtime)

---

## Architecture (high level)

```
           +-------------------+
PDF file ->| PDF parser        |-> per-page text + tables
           +-------------------+
                     |
                     v
           +-------------------+
           | Chunker           |-> text/table chunks with metadata
           +-------------------+
                     |
                     v
           +-------------------+
           | Embeddings        |-> vectors (Ollama embedding model)
           +-------------------+
                     |
                     v
           +-------------------+
           | Vector store      |-> ChromaDB collection (cosine)
           +-------------------+
                     |
                     v
           +-------------------+
           | Semantic search   |-> top-K chunks per field
           | (optional rerank) |
           +-------------------+
                     |
                     v
           +-------------------+
           | Regex extractor   |-> deterministic fields (optional)
           +-------------------+
                     |
                     v
           +-------------------+
           | LLM extractor     |-> EquipmentSchema + meta
           +-------------------+
```

The “unit of retrieval” is a **chunk** (text block or table row) stored with metadata:

- `doc_id`, `page_number`, `chunk_id`
- `chunk_type`: `text` / `table` / `ocr` (depending on pipeline)

---

## Extraction pipeline

There are two entry points:

- **API**: `backend/main.py` (FastAPI)
- **CLI**: `backend/cli_test.py` (batch indexing + extraction, writes JSON outputs)

### API pipeline (streaming)
Implemented in `backend/main.py`.

1. Save uploaded PDF to `backend/uploads/`
2. Stream pages with `services/pdf_parser.py` (`iter_pages()`)
3. Chunk each page (`services/chunker.py`)
4. Embed and store chunks into a **per-request isolated Chroma collection**
5. Semantic search per field (`services/semantic_search.py`)
6. Optional regex extraction (`services/regex_extractor.py`)
7. LLM extraction (`services/llm_extractor.py`)
8. Return:
   - structured `EquipmentSchema`
   - optional evidence (top chunks per field)
   - optional meta (accept/reject checks per field)

### CLI pipeline (single-batch)
Implemented in `backend/cli_test.py`.

1. Parse & chunk all pages first
2. Embed/store
3. Semantic search per field
4. Optional regex extraction
5. LLM extraction
6. Save:
   - `<prefix>_extracted.json`
   - `<prefix>_evidence.json`

---

## Key backend modules

### `backend/config.py`
Central configuration:

- **Models**
  - `TEXT_MODEL` (LLM for extraction)
  - `EMBEDDING_MODEL` (vector embeddings)
  - Optional reranker: `ENABLE_RERANKING`, `RERANKER_MODEL`, `RERANK_CANDIDATES`, `RERANK_BATCH_SIZE`
- **Extraction policy**
  - `ACCURACY_FIRST`
  - `MIN_EXTRACT_CONFIDENCE`
- **Chunking**
  - `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Retrieval**
  - `TOP_K_CHUNKS`
- **Regex extraction toggle**
  - `ENABLE_REGEX_EXTRACTION`

### `backend/models/schema.py`
Pydantic schemas for:

- `EquipmentSchema` (final extracted data)
- `ExtractionResponse` (API response shape)
- Field descriptions + allowed values (used by retrieval/prompting)

### `backend/services/pdf_parser.py`
PDF parsing using PyMuPDF (`fitz`) with a streaming iterator:

- `open_pdf()` returns a lightweight handle
- `iter_pages()` yields `PageContent` one page at a time
- Optional table extraction via `pdfplumber` when available

### `backend/services/chunker.py`
Builds chunks from page text and tables:

- `chunk_text()` creates overlapping text chunks
- `make_table_chunks()` creates table chunks / table-row chunks

### `backend/services/embeddings.py`
Embedding helpers using Ollama:

- `generate_embedding()`
- `generate_embeddings_batch()`

### `backend/services/vector_store.py`
ChromaDB persistence and query:

- `store_chunks_batch()` embeds + inserts into Chroma
- `query_chunks()` searches by embedding distance
- Collections stored in `chroma_db/`

### `backend/services/semantic_search.py`
Per-field retrieval from Chroma:

- `search_for_field()` runs field-specific queries
- deduplicates chunk candidates
- optionally reranks when `ENABLE_RERANKING=True`

### `backend/services/regex_extractor.py`
Deterministic extractor (optional) for fields that match reliably via patterns.

Typical use:

- brand/manufacturer
- model/reference formats
- protocols (HART/Modbus/Profibus/etc.)
- protection class (IPxx)

### `backend/services/llm_extractor.py`
LLM-based extraction and validation:

- builds a per-field prompt from `FIELD_DESCRIPTIONS`
- calls Ollama chat (`TEXT_MODEL`)
- validates output with quote anchoring (value must be supported by an exact quote)
- returns both extracted values and per-field meta (accept/reject + reasons)

### `backend/services/document_classifier.py`
Lightweight document type classification (e.g., instrument datasheet vs other) to support gating.

### `backend/services/pipeline.py`
A reusable pipeline wrapper (`run_extraction_pipeline`) used by backend flows.

---

## Outputs

The CLI writes two files per run:

- `<prefix>_extracted.json`: final structured data
- `<prefix>_evidence.json`: evidence chunks + LLM meta (acceptance checks)

---

## Setup (Windows)

### 1) Python backend

From repo root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

### 2) Ollama

Install and run Ollama, then pull the models referenced in `backend/config.py`:

```powershell
ollama pull gemma3:4b
ollama pull embeddinggemma
```

(Optional) reranker:

```powershell
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
```

---

## Run

### API (FastAPI)

```powershell
python backend\main.py
```

Then:

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/extract` with a PDF file

### CLI

```powershell
python backend\cli_test.py "backend/uploads/your.pdf" --output-prefix myrun
```

---

## Frontend (optional)

The frontend is a Vite + React app in `frontend/`.

```powershell
npm install
npm run dev
```

The backend enables CORS for local dev:

- `http://localhost:5173`

---

## Notes / common tweaks

- **Too many nulls** usually means:
  - retrieval chunks are not relevant (`TOP_K_CHUNKS` too low)
  - quote validation is rejecting outputs (quote not verbatim)
  - the value is not explicitly present in the PDF

- **Better retrieval**:
  - increase `TOP_K_CHUNKS`
  - enable reranking (`ENABLE_RERANKING=True`) once stable

- **Speed vs accuracy**:
  - `LLM_MAX_WORKERS` controls parallel field extraction
  - chunk size and overlap impact both speed and recall

---

## Repository layout

- `backend/`
  - `main.py`: FastAPI app + streaming extraction endpoint
  - `cli_test.py`: CLI pipeline to index/extract and write JSON outputs
  - `config.py`: configuration knobs
  - `models/`: Pydantic schemas and field definitions
  - `services/`: pipeline components (parser/chunker/store/search/extract)
  - `uploads/`: local PDFs (dev)
- `frontend/`: Vite + React UI
- `chroma_db/`: persistent vector store data
- `poppler-25.12.0/`: local Poppler distribution (used by some PDF tooling/scripts)

---

## License

Add a license if you plan to distribute this project.
