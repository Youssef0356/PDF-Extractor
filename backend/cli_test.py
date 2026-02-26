"""
CLI Test script for the PDF Extractor.
Optimized pipeline: collect all chunks (text only, no images), then batch-embed once.

Usage (run from the backend folder):
    python cli_test.py "C:/path/to/your/document.pdf"
"""
import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TEXT_MODEL, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME
from services.pdf_parser import open_pdf, iter_pages
from services.chunker import chunk_text, Chunk
from services.embeddings import generate_embeddings_batch
from services.vector_store import clear_collection
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields


def run_extraction(file_path: str):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        print(f"Provide the full absolute path or a path relative to the backend folder.")
        return

    print(f"\n{'='*60}")
    print(f"  PDF EXTRACTOR - OPTIMIZED PIPELINE")
    print(f"  File     : {os.path.basename(file_path)}")
    print(f"  LLM      : {TEXT_MODEL}")
    print(f"  Embeddings: {EMBEDDING_MODEL}")
    print(f"{'='*60}")

    total_start = time.time()

    # -- Phase 1: Parse + Chunk (CPU only, very fast) ---------------
    t0 = time.time()
    print(f"\n[1/4] Parsing & chunking PDF...")

    pdf = open_pdf(file_path)
    print(f"  -> {pdf.total_pages} pages")

    all_chunks: list[Chunk] = []

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
            doc_id="cli",
        )
        all_chunks.extend(chunks)

    print(f"  -> {len(all_chunks)} chunks in {time.time()-t0:.1f}s")

    # -- Phase 2: Single batch embedding call -----------------------
    t0 = time.time()
    print(f"\n[2/4] Embedding {len(all_chunks)} chunks (single batch)...")

    texts = [c.text for c in all_chunks]
    embeddings = generate_embeddings_batch(texts)

    print(f"  -> Done in {time.time()-t0:.1f}s")

    # -- Phase 3: Store in ChromaDB ---------------------------------
    t0 = time.time()
    print(f"\n[3/4] Storing in ChromaDB...")

    collection = clear_collection(CHROMA_COLLECTION_NAME)
    
    batch_size = 5000
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=texts[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=[
                {
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "has_images": c.has_images,
                }
                for c in batch
            ],
        )

    print(f"  -> {len(all_chunks)} chunks stored in {time.time()-t0:.1f}s")

    # -- Phase 4: Semantic search ----------------------------------
    print(f"\n[3.5/4] Semantic search per field...")
    field_chunks = search_all_fields()

    # -- Phase 5: LLM extraction -----------------------------------
    print(f"\n[4/4] Extracting values with LLM ({TEXT_MODEL})...")
    equipment = extract_all_fields(field_chunks)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"{'='*60}")

    print("\nEXTRACTED DATA:")
    result = equipment.model_dump(exclude_none=True)
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF Extractor CLI - Optimized single-batch pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli_test.py "C:/Users/nejiy/Desktop/myfile.pdf"
    python cli_test.py "uploads/0698a3d4_6_servo e83104fr.pdf"
        """
    )
    parser.add_argument("file", help="Path to a PDF file")
    args = parser.parse_args()
    run_extraction(args.file)
