"""
PDF Extractor Backend - FastAPI Application
Main entry point for the PDF extraction pipeline (streaming version).
"""
import hashlib
import os
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import UPLOAD_DIR, CHROMA_COLLECTION_NAME
from models.schema import ExtractionResponse, EquipmentSchema
from services.pdf_parser import open_pdf, iter_pages
from services.chunker import chunk_text, make_table_chunks
from services.vector_store import clear_collection, store_chunks_batch, document_is_indexed
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields_with_meta
from services.regex_extractor import extract_with_regex


# -- FastAPI App ----------------------------------------------------
app = FastAPI(
    title="PDF Extractor API",
    description="Extract equipment information from industrial PDFs using AI",
    version="2.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "PDF Extractor API is running (streaming mode)"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_pdf(file: UploadFile = File(...)):
    """
    Streaming extraction pipeline:
    1. Save uploaded PDF
    2. Open PDF handle (no pages in RAM)
    3. Per page: extract text -> chunk -> embed -> store -> discard
    4. Semantic search per form field
    5. LLM extraction for each field
    6. Return structured JSON
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    start_time = time.time()
    file_path = None

    try:
        # -- Step 1: Save the uploaded PDF -------------------------
        content = await file.read()
        # Stable doc_id so we can detect re-uploads of the same PDF.
        doc_id = hashlib.sha256(content).hexdigest()[:12]
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(content)

        print(f"\n{'='*60}")
        print(f"  Processing: {file.filename} ({len(content)} bytes)")
        print(f"{'='*60}")

        # -- Step 2: Open PDF (streaming) --------------------------
        pdf = open_pdf(file_path)
        print(f"\n[2/5] Opened PDF: {pdf.total_pages} pages")

        # -- Step 3: Index into Chroma (ONLY if not already indexed) --
        full_text_parts: list[str] = []
        if document_is_indexed(doc_id, CHROMA_COLLECTION_NAME):
            print(f"\n[3/5] Skipping indexing: doc_id {doc_id} already in Chroma")
        else:
            print(f"\n[3/5] Streaming pages (chunk + embed + store)...")
            collection = clear_collection(CHROMA_COLLECTION_NAME)
            total_chunks = 0
            chunk_index_counter = 0

            for page in iter_pages(pdf):
                if not page.text.strip():
                    continue

                # Keep full_text for regex/LLM extraction free of table artifacts.
                full_text_parts.append(page.text)

                text_chunks = chunk_text(
                    text=page.text,
                    page_number=page.page_number,
                    has_images=page.has_images,
                    doc_id=doc_id,
                )

                table_chunks: list = []
                if getattr(page, "tables_structured", None):
                    for ti, t in enumerate(page.tables_structured):
                        table_chunks.extend(
                            make_table_chunks(
                                doc_id=doc_id,
                                page_number=page.page_number,
                                table_index=ti,
                                table_markdown=t.get("markdown", ""),
                                row_texts=t.get("row_texts", []) or [],
                                chunk_index_start=chunk_index_counter + 1000,
                            )
                        )
                elif page.tables:
                    # Fallback: store existing table-like strings as table chunks.
                    for ti, table_text in enumerate(page.tables):
                        table_chunks.extend(
                            make_table_chunks(
                                doc_id=doc_id,
                                page_number=page.page_number,
                                table_index=ti,
                                table_markdown=("[TABLE DATA]\n" + table_text),
                                row_texts=[],
                                chunk_index_start=chunk_index_counter + 1000,
                            )
                        )

                chunks = text_chunks + table_chunks

                if chunks:
                    store_chunks_batch(chunks, collection)
                    total_chunks += len(chunks)
                    chunk_index_counter += 1

            print(f"  -> {total_chunks} chunks indexed")

        # -- Step 4: Semantic search --------------------------------
        print(f"\n[4/5] Semantic search...")
        field_chunks = search_all_fields(doc_id=doc_id)

        # -- Step 5: LLM extraction --------------------------------
        print(f"\n[5/5] Extracting with LLM...")
        full_text = "\n\n".join(full_text_parts)
        if not full_text.strip():
            full_text = "\n\n".join([(p.text or "") for p in iter_pages(pdf)])
        regex_results = extract_with_regex(full_text)
        equipment, _meta = extract_all_fields_with_meta(field_chunks, regex_results=regex_results)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Done in {elapsed:.1f}s")
        print(f"{'='*60}")

        try:
            os.remove(file_path)
        except Exception:
            pass

        return ExtractionResponse(
            success=True,
            data=equipment,
            message=f"Extracted from {pdf.total_pages} pages in {elapsed:.1f}s",
            processing_time_seconds=round(elapsed, 2),
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] {e}")

        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        return ExtractionResponse(
            success=False,
            message=f"Error processing PDF: {str(e)}",
            processing_time_seconds=round(elapsed, 2),
        )
