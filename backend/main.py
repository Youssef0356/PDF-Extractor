"""
PDF Extractor Backend - FastAPI Application
Main entry point for the PDF extraction pipeline (streaming version).
"""
import os
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import UPLOAD_DIR, CHROMA_COLLECTION_NAME
from models.schema import ExtractionResponse, EquipmentSchema
from services.pdf_parser import open_pdf, iter_pages
from services.chunker import chunk_text
from services.vector_store import clear_collection, store_chunks_batch, query_chunks
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields


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
        doc_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        print(f"\n{'='*60}")
        print(f"  Processing: {file.filename} ({len(content)} bytes)")
        print(f"{'='*60}")

        # -- Step 2: Open PDF (streaming) --------------------------
        pdf = open_pdf(file_path)
        print(f"\n[2/5] Opened PDF: {pdf.total_pages} pages")

        # -- Step 3: Stream -> chunk -> embed -> store -------------
        print(f"\n[3/5] Streaming pages (chunk + embed + store)...")
        collection = clear_collection(CHROMA_COLLECTION_NAME)
        total_chunks = 0

        for page in iter_pages(pdf):
            if not page.text.strip():
                continue

            combined = page.text
            if page.tables:
                combined += "\n\n[TABLE DATA]\n" + "\n".join(page.tables)

            chunks = chunk_text(
                text=combined,
                page_number=page.page_number,
                has_images=page.has_images,
                doc_id=doc_id,
            )

            if chunks:
                store_chunks_batch(chunks, collection)
                total_chunks += len(chunks)

        print(f"  -> {total_chunks} chunks indexed")

        # -- Step 4: Semantic search --------------------------------
        print(f"\n[4/5] Semantic search...")
        field_chunks = search_all_fields()

        # -- Step 5: LLM extraction --------------------------------
        print(f"\n[5/5] Extracting with LLM...")
        equipment = extract_all_fields(field_chunks)

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
