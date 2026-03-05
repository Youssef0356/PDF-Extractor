import os
import time
from typing import Tuple

from config import (
    CHROMA_COLLECTION_NAME,
    TABLE_CHUNK_OFFSET,
    ENABLE_REGEX_EXTRACTION,
    MIN_EXTRACT_CONFIDENCE,
    logger,
)
from models.schema import EquipmentSchema
from services.pdf_parser import open_pdf, iter_pages, render_page_image
from services.chunker import chunk_text, make_table_chunks
from services.vector_store import clear_collection, store_chunks_batch, document_is_indexed
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields_with_meta
from services.regex_extractor import extract_with_regex
from services.document_classifier import classify_document


def run_extraction_pipeline(file_path: str, doc_id: str) -> Tuple[EquipmentSchema, float, int]:
    """
    Core extraction pipeline:
    1. Open PDF handle (no pages in RAM)
    2. Per page: extract text -> chunk -> embed -> store -> discard
    3. Semantic search per form field
    4. LLM extraction for each field
    5. Return structured JSON
    """
    start_time = time.time()

    pdf = open_pdf(file_path)
    logger.info(f"Opened PDF: {pdf.total_pages} pages")

    full_text_parts: list[str] = []

    if document_is_indexed(doc_id, CHROMA_COLLECTION_NAME):
        logger.info(f"Skipping indexing: doc_id {doc_id} already in Chroma")
    else:
        logger.info(f"Streaming pages (chunk + embed + store)...")
        collection = clear_collection(CHROMA_COLLECTION_NAME)
        total_chunks = 0
        chunk_index_counter = 0
        ocr_pages_done = 0

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
                            chunk_index_start=chunk_index_counter + TABLE_CHUNK_OFFSET,
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
                            chunk_index_start=chunk_index_counter + TABLE_CHUNK_OFFSET,
                        )
                    )

            chunks = text_chunks + table_chunks

            if chunks:
                store_chunks_batch(chunks, collection)
                total_chunks += len(chunks)
                chunk_index_counter += 1

        logger.info(f"Indexed {total_chunks} chunks")

    logger.info("Semantic search...")
    field_chunks = search_all_fields(doc_id=doc_id)

    logger.info("Extracting with LLM...")
    full_text = "\n\n".join(full_text_parts)
    if not full_text.strip():
        full_text = "\n\n".join([(p.text or "") for p in iter_pages(pdf)])
    
    regex_results = {}
    if ENABLE_REGEX_EXTRACTION:
        logger.info("Running deterministic regex extraction...")
        regex_results = extract_with_regex(full_text)
    else:
        logger.info("Regex extraction disabled via config. Skipping...")

    doc_ctx = classify_document(full_text)
    
    logger.info(f"DOC doc_type={doc_ctx.doc_type} conf={doc_ctx.confidence:.2f} ({doc_ctx.rationale})")
    
    equipment, _meta = extract_all_fields_with_meta(
        field_chunks,
        min_confidence=MIN_EXTRACT_CONFIDENCE,
        regex_results=regex_results,
        doc_ctx=doc_ctx,
    )

    elapsed = time.time() - start_time
    logger.info(f"Extraction Pipeline Done in {elapsed:.1f}s")

    return equipment, elapsed, pdf.total_pages
