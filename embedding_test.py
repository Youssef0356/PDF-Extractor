#!/usr/bin/env python3
"""
PDF Text Embedding and Search Script
Uses embedding models to store PDF text and enables semantic search
No OCR, no frontend - just terminal-based embedding and search
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the backend directory to the path so we can import services
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.pdf_parser import open_pdf, iter_pages
from services.chunker import chunk_text
from services.embeddings import generate_embedding, generate_embeddings_batch
from services.vector_store import clear_collection, store_chunks_batch, query_chunks

# ===== CONFIGURATION =====
# Change this variable to point to your PDF file
PDF_FILE_PATH = "Non-essential Files/6_vanne e80150fr.pdf"  # <- MODIFY THIS LINE

# Schema (fields you want to extract later)
SCHEMA_PATH = "JsonFormatting.json"  # <- MODIFY THIS LINE if needed

# Embedding and search configuration
COLLECTION_NAME = "pdf_text_collection"
CHUNK_SIZE = 500  # Smaller chunks for better embedding
CHUNK_OVERLAP = 50

# Retrieval tuning
TOP_K = 3
# If set (e.g. 0.35), results with distance > MAX_DISTANCE will be treated as "no good match".
# Set to None to disable.
MAX_DISTANCE = None

# Fields to test (if empty, will be loaded from SCHEMA_PATH)
TEST_FIELDS: list[str] = []

# Field -> natural language query templates / synonyms.
# This is the critical piece: you must query with wording the PDF is likely to contain.
FIELD_QUERIES: dict[str, list[str]] = {
}


def interactive_search_loop() -> None:
    while True:
        query = input("\nSearch query (empty to quit): ").strip()
        if not query:
            break

        results = query_chunks(
            query_text=query,
            n_results=TOP_K,
            collection_name=COLLECTION_NAME,
        )

        if MAX_DISTANCE is not None:
            results = [r for r in results if r.get("distance", 1.0) <= MAX_DISTANCE]

        if not results:
            if MAX_DISTANCE is None:
                print("No results")
            else:
                print(f"No results under MAX_DISTANCE={MAX_DISTANCE}")
            continue

        print("\nTop matches:")
        for i, r in enumerate(results, start=1):
            page = r.get("metadata", {}).get("page_number", "Unknown")
            dist = r.get("distance", 0.0)
            text = r.get("text", "")
            text_preview = text.replace("\n", " ")
            print(f"  {i}. Page {page} | distance={dist:.4f} | {text_preview[:220]}...")


def _load_schema_fields(schema_path: str) -> list[str]:
    if not os.path.exists(schema_path):
        return []
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.keys())
    return []


def _build_queries_for_field(field_name: str) -> list[str]:
    # Prefer curated synonyms, fallback to a generic question.
    if field_name in FIELD_QUERIES:
        return FIELD_QUERIES[field_name]
    return [
        f"Quelle est la valeur pour '{field_name}' ?",
        f"'{field_name}'",
    ]

# ===== EMBEDDING FUNCTIONS =====

def create_chunks_for_embedding(pdf_path: str) -> list:
    """Create optimized chunks for embedding."""
    print(f"Creating chunks from: {pdf_path}")
    
    pdf = open_pdf(pdf_path)
    all_chunks = []
    
    for page in iter_pages(pdf):
        if not page.text.strip():
            continue
            
        # Combine text and tables
        combined = page.text
        if page.tables:
            combined += "\n\n[TABLE DATA]\n" + "\n".join(page.tables)
        
        # Create smaller chunks optimized for embedding
        page_chunks = chunk_text(
            text=combined,
            page_number=page.page_number,
            has_images=False,
            doc_id="pdf_doc"
        )
        
        all_chunks.extend(page_chunks)
    
    return all_chunks

def embed_and_store(chunks: list, collection_name: str):
    """Embed chunks and store in vector database."""
    print(f"Embedding {len(chunks)} chunks...")
    
    # Clear existing collection
    collection = clear_collection(collection_name)
    
    # Store chunks in batches
    store_chunks_batch(chunks, collection)
    
    print(f"Successfully stored {len(chunks)} chunks in vector database")
    return collection

def test_semantic_search(collection, queries: list) -> dict:
    """Test semantic search with various queries."""
    print("\n" + "="*60)
    print("TESTING SEMANTIC SEARCH")
    print("="*60)
    
    results = {}
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        try:
            # Search for relevant chunks
            search_results = query_chunks(
                query_text=query,
                n_results=TOP_K,
                collection_name=COLLECTION_NAME,
            )

            # Optional: filter out weak matches
            if MAX_DISTANCE is not None:
                search_results = [r for r in search_results if r.get("distance", 1.0) <= MAX_DISTANCE]

            if search_results:
                results[query] = {
                    "found": True,
                    "results": [],
                }

                for i, result in enumerate(search_results):
                    page = result.get("metadata", {}).get("page_number", "Unknown")
                    dist = result.get("distance", 0)
                    print(f"  Result {i+1} (Page {page}, distance={dist:.4f}):")
                    print(f"    {result['text'][:150]}...")
                    print()

                    results[query]["results"].append({
                        "page": page,
                        "chunk_id": result.get("metadata", {}).get("chunk_id"),
                        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                        "distance": dist,
                    })
            else:
                if MAX_DISTANCE is None:
                    print("  No results found")
                else:
                    print(f"  No results under MAX_DISTANCE={MAX_DISTANCE}")
                results[query] = {"found": False}

        except Exception as e:
            print(f"  Error: {e}")
            results[query] = {"found": False, "error": str(e)}
    
    return results

def analyze_embedding_quality(chunks: list, collection):
    """Analyze the quality of embeddings and search results."""
    print("\n" + "="*60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*60)
    
    # Test with some sample text from the chunks
    sample_texts = [chunk.text for chunk in chunks[:5]]
    
    for i, sample_text in enumerate(sample_texts):
        print(f"\nSample {i+1}:")
        print(f"Text: {sample_text[:100]}...")
        
        # Use first few words as query
        query_words = sample_text.split()[:5]
        query = " ".join(query_words)
        
        print(f"Query: '{query}'")
        
        try:
            search_results = query_chunks(
                query_text=query,
                n_results=2,
                collection_name=COLLECTION_NAME,
            )
            if search_results:
                print("✅ Found matching chunks")
                for j, result in enumerate(search_results):
                    dist = result.get("distance", 0)
                    print(f"  Match {j+1} (distance={dist:.4f}): {result['text'][:80]}...")
            else:
                print("❌ No matches found")
        except Exception as e:
            print(f"❌ Error: {e}")

# ===== MAIN FUNCTION =====

def main():
    """Main embedding and search function."""
    print("=" * 60)
    print("PDF TEXT EMBEDDING AND SEARCH TEST")
    print("=" * 60)
    
    # Check if PDF file exists
    if not os.path.exists(PDF_FILE_PATH):
        print(f"ERROR: PDF file not found: {PDF_FILE_PATH}")
        print("Please modify the PDF_FILE_PATH variable in this script")
        return
    
    start_time = time.time()
    
    try:
        # Step 1: Create chunks
        print("\n[1/4] Creating text chunks...")
        chunks = create_chunks_for_embedding(PDF_FILE_PATH)
        print(f"Created {len(chunks)} chunks")
        
        # Step 2: Embed and store
        print("\n[2/4] Creating embeddings and storing in vector database...")
        collection = embed_and_store(chunks, COLLECTION_NAME)
        
        # Step 3: Test semantic search
        print("\n[3/4] Search mode...")
        mode = input("Choose mode: (i)nteractive search or (b)atch field test? [i/b]: ").strip().lower() or "i"

        search_results_by_field: dict[str, dict] = {}
        fields: list[str] = []

        if mode.startswith("b"):
            schema_fields = _load_schema_fields(SCHEMA_PATH)
            fields = TEST_FIELDS if TEST_FIELDS else schema_fields
            if not fields:
                fields = [
                    "type_equipement",
                    "modele",
                    "marque",
                    "signal",
                    "communication",
                ]

            field_to_queries: dict[str, list[str]] = {f: _build_queries_for_field(f) for f in fields}
            for field, queries in field_to_queries.items():
                print("\n" + "=" * 60)
                print(f"FIELD: {field}")
                print("=" * 60)
                search_results_by_field[field] = {
                    "queries": queries,
                    "results_by_query": test_semantic_search(collection, queries),
                }
        else:
            interactive_search_loop()
        
        # Step 4: Analyze embedding quality
        print("\n[4/4] Analyzing embedding quality...")
        analyze_embedding_quality(chunks, collection)
        
        # Save results
        output_data = {
            'source_file': PDF_FILE_PATH,
            'total_chunks': len(chunks),
            'collection_name': COLLECTION_NAME,
            'processing_time_seconds': round(time.time() - start_time, 2),
            'schema_path': SCHEMA_PATH,
            'test_fields': fields,
            'top_k': TOP_K,
            'max_distance': MAX_DISTANCE,
            'search_results_by_field': search_results_by_field,
            'chunk_sample': [
                {
                    'id': chunk.chunk_id,
                    'page': chunk.page_number,
                    'text_length': len(chunk.text),
                    'preview': chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
                }
                for chunk in chunks[:5]
            ]
        }
        
        with open('embedding_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✅ Processed {len(chunks)} chunks from PDF")
        print(f"✅ Stored embeddings in collection: {COLLECTION_NAME}")
        if fields:
            print(f"✅ Tested {len(fields)} fields")
        print(f"✅ Processing time: {time.time() - start_time:.2f} seconds")
        print(f"✅ Results saved to: embedding_test_results.json")
        
        # Show successful searches
        if not search_results_by_field:
            return
        all_queries = []
        for field_data in search_results_by_field.values():
            all_queries.extend(list(field_data.get("results_by_query", {}).keys()))
        # Count successful queries
        successful_queries = 0
        total_queries = 0
        for field_data in search_results_by_field.values():
            for q, qres in field_data.get("results_by_query", {}).items():
                total_queries += 1
                if qres.get("found"):
                    successful_queries += 1
        print(f"✅ Successful queries: {successful_queries}/{total_queries}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
