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
from services.chunker import chunk_text, Chunk, make_table_chunks
from services.pdf_parser import open_pdf, iter_pages
from services.embeddings import generate_embeddings_batch
from services.vector_store import clear_collection, query_chunks, store_chunks_batch
from services.semantic_search import search_all_fields
from services.llm_extractor import extract_all_fields_with_meta
from services.regex_extractor import extract_with_regex


def _index_pdf_to_chroma(file_path: str, collection_name: str) -> int:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        print(f"Provide the full absolute path or a path relative to the backend folder.")
        return 0

    total_start = time.time()

    # -- Phase 1: Parse + Chunk (CPU only, very fast) ---------------
    t0 = time.time()
    print(f"\n[1/3] Parsing & chunking PDF...")

    pdf = open_pdf(file_path)
    print(f"  -> {pdf.total_pages} pages")

    all_chunks: list[Chunk] = []
    all_page_texts: list[str] = []
    chunk_index_counter = 0

    for page in iter_pages(pdf):
        if not page.text.strip():
            continue

        # Keep regex/LLM full_text free of table artifacts.
        all_page_texts.append(page.text)

        text_chunks = chunk_text(
            text=page.text,
            page_number=page.page_number,
            has_images=False,
            doc_id="cli",
        )

        table_chunks: list[Chunk] = []
        if getattr(page, "tables_structured", None):
            for ti, t in enumerate(page.tables_structured):
                table_chunks.extend(
                    make_table_chunks(
                        doc_id="cli",
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
                        doc_id="cli",
                        page_number=page.page_number,
                        table_index=ti,
                        table_markdown=("[TABLE DATA]\n" + table_text),
                        row_texts=[],
                        chunk_index_start=chunk_index_counter + 1000,
                    )
                )

        all_chunks.extend(text_chunks + table_chunks)
        chunk_index_counter += 1

    print(f"  -> {len(all_chunks)} chunks in {time.time()-t0:.1f}s")

    if not all_chunks:
        print("[WARN] No text chunks found. PDF might be image-based or empty.")
        return 0

    # -- Phase 2+3: Embed + Store (uses vector_store to preserve metadata) ---
    t0 = time.time()
    print(f"\n[2/3] Embedding {len(all_chunks)} chunks (single batch)...")
    print(f"  -> Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"\n[3/3] Storing in ChromaDB collection '{collection_name}'...")
    collection = clear_collection(collection_name)
    store_chunks_batch(all_chunks, collection)
    print(f"  -> {len(all_chunks)} chunks stored in {time.time()-t0:.1f}s")

    elapsed = time.time() - total_start
    print(f"\n[Index] DONE in {elapsed:.1f}s")

    full_text = "\n\n".join(all_page_texts)
    return len(all_chunks), full_text


def _print_query_results(query: str, results: list[dict], max_distance: float | None) -> None:
    print(f"\nQuery: {query}")
    if max_distance is not None:
        results = [r for r in results if r.get("distance", 1.0) <= max_distance]
    if not results:
        if max_distance is None:
            print("No results")
        else:
            print(f"No results under max_distance={max_distance}")
        return

    print("Top matches:")
    for i, r in enumerate(results, start=1):
        page = r.get("metadata", {}).get("page_number", "Unknown")
        dist = r.get("distance", 0.0)
        text = (r.get("text") or "").replace("\n", " ")
        print(f"  {i}. Page {page} | distance={dist:.4f} | {text[:220]}...")


def _interactive_search(collection_name: str, top_k: int, max_distance: float | None) -> None:
    print("\nInteractive semantic search (empty query to quit)")
    while True:
        q = input("Search query: ").strip()
        if not q:
            break
        results = query_chunks(query_text=q, n_results=top_k, collection_name=collection_name)
        _print_query_results(q, results, max_distance)


def _filter_field_chunks_by_distance(
    field_chunks: dict[str, list[dict]],
    max_distance: float | None,
) -> dict[str, list[dict]]:
    if max_distance is None:
        return field_chunks

    filtered: dict[str, list[dict]] = {}
    for field, chunks in field_chunks.items():
        filtered[field] = [c for c in chunks if c.get("distance", 1.0) <= max_distance]
    return filtered


def _build_evidence(field_chunks: dict[str, list[dict]]) -> dict:
    evidence: dict[str, list[dict]] = {}
    for field, chunks in field_chunks.items():
        evidence[field] = [
            {
                "page_number": (c.get("metadata") or {}).get("page_number"),
                "chunk_id": (c.get("metadata") or {}).get("chunk_id"),
                "distance": c.get("distance"),
                "text_preview": (c.get("text") or "")[:300],
            }
            for c in chunks
        ]
    return evidence


def run_extraction(
    file_path: str,
    collection_name: str,
    no_llm: bool,
    field_max_distance: float | None,
    min_confidence: float,
    output_prefix: str,
):
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

    # Index always (this is your embedding test integrated in backend)
    result = _index_pdf_to_chroma(file_path, collection_name)
    if isinstance(result, tuple):
        count, full_text = result
    else:
        count, full_text = result, ""
    if count == 0:
        return

    if no_llm:
        elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  DONE (index only) in {elapsed:.1f}s")
        print(f"{'='*60}")
        return

    # Regex extraction (fast, deterministic)
    print(f"\n[4/6] Regex extraction (deterministic fields)...")
    regex_results = extract_with_regex(full_text)
    if regex_results:
        print(f"  -> {len(regex_results)} fields matched by regex")
    else:
        print(f"  -> No regex matches")

    # Semantic search per field + LLM extraction (kept for later)
    print(f"\n[5/6] Semantic search per field...")
    field_chunks = search_all_fields()
    field_chunks = _filter_field_chunks_by_distance(field_chunks, field_max_distance)
    evidence = _build_evidence(field_chunks)

    print(f"\n[6/6] Extracting values with LLM ({TEXT_MODEL}) + regex...")
    equipment, llm_meta = extract_all_fields_with_meta(
        field_chunks, min_confidence=min_confidence, regex_results=regex_results
    )

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"{'='*60}")

    print("\nEXTRACTED DATA:")
    full_result = equipment.model_dump(exclude_none=False)

    result = full_result
    print(json.dumps(result, indent=4, ensure_ascii=False))

    extracted_path = f"{output_prefix}_extracted.json"
    evidence_path = f"{output_prefix}_evidence.json"
    with open(extracted_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "file": file_path,
                "collection": collection_name,
                "field_max_distance": field_max_distance,
                "min_confidence": min_confidence,
                "evidence": evidence,
                "llm_meta": llm_meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved: {extracted_path}")
    print(f"Saved: {evidence_path}")


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

    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {CHROMA_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only parse/chunk/embed/store (no LLM)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Same as --index-only (kept for clarity)",
    )
    parser.add_argument(
        "--search",
        default="",
        help="Run a semantic search query after indexing (prints top matches)",
    )
    parser.add_argument(
        "--interactive-search",
        action="store_true",
        help="After indexing, enter an interactive search loop",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to return for search queries",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Optional: filter out results with distance > max_distance",
    )

    parser.add_argument(
        "--field-max-distance",
        type=float,
        default=None,
        help="Optional: filter semantic-search chunks per field before sending them to the LLM",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence for accepting an extracted field (LLM-side default is 0.3)",
    )
    parser.add_argument(
        "--output-prefix",
        default="cli_output",
        help="Prefix for output files (writes <prefix>_extracted.json and <prefix>_evidence.json)",
    )

    args = parser.parse_args()

    no_llm = bool(args.index_only or args.no_llm)
    run_extraction(
        args.file,
        args.collection,
        no_llm=no_llm,
        field_max_distance=args.field_max_distance,
        min_confidence=args.min_confidence,
        output_prefix=args.output_prefix,
    )

    if args.search:
        results = query_chunks(
            query_text=args.search,
            n_results=args.top_k,
            collection_name=args.collection,
        )
        _print_query_results(args.search, results, args.max_distance)

    if args.interactive_search:
        _interactive_search(args.collection, args.top_k, args.max_distance)
