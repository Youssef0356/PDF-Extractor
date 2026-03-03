#!/usr/bin/env python3
"""
Clear ChromaDB Collections
Script to clear old embedding data from the vector database
"""

import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.vector_store import clear_collection, get_or_create_collection

def clear_all_collections():
    """Clear all collections in the database."""
    print("Clearing all collections...")
    
    # Clear the default collection
    collection1 = clear_collection("pdf_chunks")
    print("✅ Cleared 'pdf_chunks' collection")
    
    # Clear the test collection
    collection2 = clear_collection("pdf_text_collection")
    print("✅ Cleared 'pdf_text_collection' collection")
    
    print("\nAll collections cleared successfully!")

def clear_specific_collection(collection_name):
    """Clear a specific collection."""
    print(f"Clearing collection: {collection_name}")
    
    try:
        collection = clear_collection(collection_name)
        print(f"✅ Cleared '{collection_name}' collection")
        return collection
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")
        return None

def list_collections():
    """List all available collections."""
    try:
        from services.vector_store import _client
        collections = _client.list_collections()
        
        print("Available collections:")
        for collection in collections:
            print(f"  - {collection.name}")
        
        return collections
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

if __name__ == "__main__":
    print("=" * 50)
    print("CHROMADB COLLECTION MANAGER")
    print("=" * 50)
    
    print("\n1. Listing current collections:")
    list_collections()
    
    print("\n2. Clearing all collections:")
    clear_all_collections()
    
    print("\n3. Verifying collections are cleared:")
    list_collections()
    
    print("\n✅ Database cleanup complete!")
