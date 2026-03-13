"""PDF Extractor Backend - FastAPI Application."""

import json
import os
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import (
    UPLOAD_DIR,
    CHROMA_COLLECTION_NAME,
    TEXT_MODEL,
    EMBEDDING_MODEL,
    TOP_K_CHUNKS,
    MIN_EXTRACT_CONFIDENCE,
    ENABLE_REGEX_EXTRACTION,
)
from models.schema import ExtractionResponse
from services.pdf_parser import open_pdf, iter_pages
from services.chunker import chunk_text, make_table_chunks
from services.vector_store import clear_collection, delete_collection, store_chunks_batch
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields_with_meta
from services.regex_extractor import extract_with_regex
from services.document_classifier import classify_document
from services.correction_store import save_correction


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
    return {"status": "ok", "message": "PDF Extractor API"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "text_model": TEXT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/feedback")
async def receive_feedback(payload: dict):
    """
    Called on 'Enregistrer l'équipement'.
    Receives list of corrections where AI was wrong.
    """
    corrections = payload.get("corrections", []) or []
    doc_type = payload.get("doc_type", "") or ""

    saved = 0
    for c in corrections:
        if not isinstance(c, dict):
            continue
        field = c.get("field")
        ai_value = c.get("ai_value")
        correct_value = c.get("correct_value")
        if field and ai_value != correct_value:
            if save_correction(str(field), "" if ai_value is None else str(ai_value), "" if correct_value is None else str(correct_value), doc_type):
                saved += 1

    return {"status": "ok", "saved": saved}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_pdf(
    file: UploadFile = File(...),
    min_confidence: float = MIN_EXTRACT_CONFIDENCE,
    top_k_chunks: int = TOP_K_CHUNKS,
    field_max_distance: float | None = None,
    enable_regex: bool = ENABLE_REGEX_EXTRACTION,
    return_evidence: bool = True,
    return_meta: bool = True,
):
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

    t_start = time.time()
    file_path = None
    collection_name = None
    doc_id = None

    try:
        # -- Step 1: Save the uploaded PDF -------------------------
        t0 = time.time()
        content = await file.read()
        doc_id = uuid.uuid4().hex[:12]
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(content)
        t_save_ms = (time.time() - t0) * 1000.0

        print(f"\n{'='*60}")
        print(f"  Processing: {file.filename} ({len(content)} bytes)")
        print(f"{'='*60}")

        # -- Step 2: Open PDF (streaming) --------------------------
        t0 = time.time()
        pdf = open_pdf(file_path)
        t_open_ms = (time.time() - t0) * 1000.0

        # -- Step 3: Index into Chroma (per-request isolated collection) --
        t0 = time.time()
        full_text_parts: list[str] = []
        collection_name = f"{CHROMA_COLLECTION_NAME}_{doc_id}"
        collection = clear_collection(collection_name)
        total_chunks = 0
        chunk_index_counter = 0

        for page in iter_pages(pdf):
            page_text = (page.text or "").strip()
            if not page_text:
                continue

            full_text_parts.append(page_text)

            text_chunks = chunk_text(
                text=page_text,
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

        t_index_ms = (time.time() - t0) * 1000.0

        # -- Step 4: Semantic search --------------------------------
        t0 = time.time()
        field_chunks = search_all_fields(
            doc_id=doc_id,
            collection_name=collection_name,
            n_results=top_k_chunks,
        )
        if field_max_distance is not None:
            filtered = {}
            for field_name, chunks in field_chunks.items():
                filtered[field_name] = [c for c in (chunks or []) if c.get("distance", 1.0) <= field_max_distance]
            field_chunks = filtered
        t_retrieve_ms = (time.time() - t0) * 1000.0

        evidence = None
        if return_evidence:
            evidence = {}
            for field_name, chunks in field_chunks.items():
                evidence[field_name] = [
                    {
                        "page_number": (c.get("metadata") or {}).get("page_number"),
                        "chunk_id": (c.get("metadata") or {}).get("chunk_id"),
                        "distance": c.get("distance"),
                        "text_preview": (c.get("text") or "")[:300],
                    }
                    for c in (chunks or [])
                ]

        # -- Step 5: LLM extraction --------------------------------
        t0 = time.time()
        full_text = "\n\n".join(full_text_parts)
        if not full_text.strip():
            full_text = "\n\n".join([(p.text or "") for p in iter_pages(pdf)])

        regex_results = {}
        if enable_regex:
            regex_results = extract_with_regex(full_text)

        doc_ctx = classify_document(full_text)
        equipment, meta = extract_all_fields_with_meta(
            field_chunks,
            min_confidence=min_confidence,
            regex_results=regex_results,
            doc_ctx=doc_ctx,
            collection_name=collection_name,
        )
        t_llm_ms = (time.time() - t0) * 1000.0

        confidence_map = None
        try:
            if isinstance(meta, dict):
                confidence_map = {
                    k: float((v or {}).get("confidence") or 0.0)
                    for k, v in meta.items()
                    if isinstance(v, dict) and (v.get("confidence") is not None)
                }
        except Exception:
            confidence_map = None

        timings_ms = {
            "save": round(t_save_ms, 2),
            "open_pdf": round(t_open_ms, 2),
            "index": round(t_index_ms, 2),
            "retrieve": round(t_retrieve_ms, 2),
            "llm": round(t_llm_ms, 2),
            "total": round((time.time() - t_start) * 1000.0, 2),
        }

        doc_context = {
            "doc_id": doc_id,
            "doc_type": getattr(doc_ctx, "doc_type", None),
            "confidence": float(getattr(doc_ctx, "confidence", 0.0) or 0.0),
            "rationale": getattr(doc_ctx, "rationale", None),
        }

        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        try:
            if collection_name:
                delete_collection(collection_name)
        except Exception:
            pass

        # Persist results to disk (same behavior as CLI outputs)
        try:
            outputs_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs"))
            os.makedirs(outputs_dir, exist_ok=True)

            safe_name = (file.filename or "document.pdf").replace("/", "_").replace("\\", "_")
            base = f"{doc_id}_{safe_name}"
            extracted_path = os.path.join(outputs_dir, f"{base}_extracted.json")
            evidence_path = os.path.join(outputs_dir, f"{base}_evidence.json")

            with open(extracted_path, "w", encoding="utf-8") as f:
                json.dump(equipment.model_dump(exclude_none=False), f, indent=2, ensure_ascii=False)

            with open(evidence_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "file": file_path,
                        "collection": collection_name,
                        "field_max_distance": field_max_distance,
                        "min_confidence": min_confidence,
                        "evidence": evidence,
                        "llm_meta": meta,
                        "confidence": confidence_map,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"[OUTPUTS] Saved extracted JSON: {extracted_path}")
            print(f"[OUTPUTS] Saved evidence JSON:  {evidence_path}")
        except Exception as e:
            print(f"[OUTPUTS] Failed to save JSON outputs: {e}")

        return ExtractionResponse(
            success=True,
            data=equipment,
            confidence=confidence_map,
            message=f"Extracted from {pdf.total_pages} pages",
            processing_time_seconds=round(time.time() - t_start, 2),
            meta=(meta if return_meta else None),
            evidence=evidence,
            doc_context=doc_context,
            timings_ms=timings_ms,
        )

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"\n[ERROR] {e}")

        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        try:
            if collection_name:
                delete_collection(collection_name)
        except Exception:
            pass

        return ExtractionResponse(
            success=False,
            message=f"Error processing PDF: {str(e)}",
            processing_time_seconds=round(elapsed, 2),
        )

import uvicorn


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
