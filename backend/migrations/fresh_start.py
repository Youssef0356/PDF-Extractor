"""
Migration script for Schema V2.
Wipes old ChromaDB collections and logs to start fresh.
"""

import os
import shutil
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME

def fresh_start():
    print("--- Starting Migration: Fresh Start (V2) ---")
    
    # 1. Wipe ChromaDB
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting ChromaDB at {CHROMA_PATH}...")
        # Chromadb client check
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = client.list_collections()
            for col in collections:
                print(f"  Dropping collection: {col.name}")
                client.delete_collection(col.name)
        except Exception as e:
            print(f"  Chroma client wipe failed: {e}. Falling back to folder deletion.")
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    
    # 2. Clear corrections
    from services.feedback import CORRECTIONS_FILE
    if os.path.exists(CORRECTIONS_FILE):
        print(f"Deleting corrections log: {CORRECTIONS_FILE}")
        os.remove(CORRECTIONS_FILE)
    
    # 3. Clear temp and outputs
    for folder in ["tmp", "outputs"]:
        if os.path.exists(folder):
            print(f"Clearing {folder} folder...")
            shutil.rmtree(folder)
            os.makedirs(folder)

    print("--- Migration Complete ---")

if __name__ == "__main__":
    fresh_start()
