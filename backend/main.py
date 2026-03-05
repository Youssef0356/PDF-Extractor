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

from config import (
    UPLOAD_DIR,
    CHROMA_COLLECTION_NAME,
    ENABLE_VISION_OCR,
    VISION_OCR_ONLY_IF_NO_TEXT,
    VISION_OCR_MAX_PAGES,
    VISION_OCR_TEXT_CHAR_THRESHOLD,
    VISION_OCR_AUTO_MIXED_PAGES,
    VISION_OCR_TEXT_AREA_RATIO_MAX,
    VISION_OCR_MAX_IMAGE_AREA_RATIO_MIN,
    VISION_OCR_MAX_DRAWING_AREA_RATIO_MIN,
    VISION_OCR_DRAWING_COUNT_MIN,
)
from models.schema import ExtractionResponse, EquipmentSchema
from services.pdf_parser import open_pdf, iter_pages, render_page_image
from services.chunker import chunk_text, make_table_chunks
from services.vector_store import clear_collection, store_chunks_batch, document_is_indexed
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields_with_meta
from services.regex_extractor import extract_with_regex
from services.vision_ocr import extract_text_from_page_image_base64
from services.document_classifier import classify_document


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
            ocr_pages_done = 0

            for page in iter_pages(pdf):
                page_text = (page.text or "").strip()

                # Optionally OCR page images (for scanned/image-based PDFs).
                ocr_text = ""
                text_len = len(page_text)
                below_threshold = text_len < VISION_OCR_TEXT_CHAR_THRESHOLD
                auto_mixed = bool(
                    VISION_OCR_AUTO_MIXED_PAGES
                    and page.has_images
                    and (getattr(page, "text_area_ratio", 0.0) <= VISION_OCR_TEXT_AREA_RATIO_MAX)
                    and (getattr(page, "max_image_area_ratio", 0.0) >= VISION_OCR_MAX_IMAGE_AREA_RATIO_MIN)
                )
                auto_vector = bool(
                    VISION_OCR_AUTO_MIXED_PAGES
                    and (getattr(page, "text_area_ratio", 0.0) <= VISION_OCR_TEXT_AREA_RATIO_MAX)
                    and (
                        (getattr(page, "max_drawing_area_ratio", 0.0) >= VISION_OCR_MAX_DRAWING_AREA_RATIO_MIN)
                        or (getattr(page, "drawing_count", 0) >= VISION_OCR_DRAWING_COUNT_MIN)
                    )
                )
                should_ocr = bool(
                    ENABLE_VISION_OCR
                    and ocr_pages_done < VISION_OCR_MAX_PAGES
                    and (
                        (not page_text)  # scanned page
                        or auto_mixed  # mixed page with large image blocks
                        or auto_vector  # vector-drawing-dominant page (no embedded images)
                        or (not VISION_OCR_ONLY_IF_NO_TEXT and below_threshold)  # legacy threshold mode
                    )
                )
                if should_ocr:
                    try:
                        img_b64 = render_page_image(pdf.file_path, page.page_number)
                        ocr_text = extract_text_from_page_image_base64(img_b64)
                        if ocr_text.strip():
                            ocr_pages_done += 1
                    except Exception as e:
                        print(f"[OCR] Failed page {page.page_number}: {e}")

                if not page_text and not ocr_text.strip():
                    continue

                # Keep full_text for regex/LLM extraction free of table artifacts.
                if page_text:
                    full_text_parts.append(page_text)
                if ocr_text.strip():
                    full_text_parts.append("[OCR]\n" + ocr_text.strip())

                text_chunks = []
                if page_text:
                    text_chunks = chunk_text(
                        text=page_text,
                        page_number=page.page_number,
                        has_images=page.has_images,
                        doc_id=doc_id,
                    )

                ocr_chunks = []
                if ocr_text.strip():
                    ocr_chunks = chunk_text(
                        text=("[OCR]\n" + ocr_text.strip()),
                        page_number=page.page_number,
                        has_images=page.has_images,
                        doc_id=doc_id,
                    )
                    for c in ocr_chunks:
                        c.chunk_type = "ocr"
                        c.chunk_id = f"{c.chunk_id}_ocr"

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

                chunks = text_chunks + ocr_chunks + table_chunks

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
        doc_ctx = classify_document(full_text)
        print(f"[DOC] doc_type={doc_ctx.doc_type} conf={doc_ctx.confidence:.2f} ({doc_ctx.rationale})")
        equipment, _meta = extract_all_fields_with_meta(field_chunks, regex_results=regex_results, doc_ctx=doc_ctx)

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
