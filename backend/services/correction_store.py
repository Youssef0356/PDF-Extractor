"""
Correction memory store.

Persists user corrections to a JSON log and a dedicated ChromaDB collection,
so future extractions can inject "past mistakes" into prompts.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from services.vector_store import get_or_create_collection
from services.embeddings import generate_embedding


CORRECTIONS_COLLECTION = "field_corrections"


def _corrections_file_path() -> str:
    # Absolute path to backend/corrections.json regardless of CWD.
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(backend_dir, "corrections.json")


def _read_json_list(path: str) -> list[dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_json_list(path: str, entries: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _correction_already_exists(collection, field: str, ai_value: str, correct_value: str) -> bool:
    try:
        results = collection.get(
            where={"$and": [{"field": field}, {"ai_value": ai_value}, {"correct_value": correct_value}]},
            include=["ids"],
        )
        return len(results.get("ids") or []) > 0
    except Exception:
        return False


def save_correction(field: str, ai_value: str, correct_value: str, doc_type: str = "", rule: str = "") -> str | None:
    """Save one correction to JSON log + embed it in ChromaDB.

    If a manual 'rule' is provided, it is used as the primary text for embedding.
    Returns the Chroma ID if stored successfully, else None.
    """
    if not field:
        return None

    ai_value = "" if ai_value is None else str(ai_value)
    correct_value = "" if correct_value is None else str(correct_value)
    if ai_value == correct_value:
        return None

    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "field": field,
        "ai_value": ai_value,
        "correct_value": correct_value,
        "doc_type": doc_type or "",
        "rule": rule or "",
        "timestamp": ts,
    }

    if rule:
        text = f'RULE for field "{field}": {rule} (Value should be "{correct_value}", not "{ai_value}")'
    else:
        text = (
            f'Field "{field}": AI said "{ai_value}" but correct value is "{correct_value}". '
            f'Doc type: {doc_type or "unknown"}.'
        )
    
    uid = f"corr_{uuid.uuid4().hex}"
    try:
        collection = get_or_create_collection(CORRECTIONS_COLLECTION)
        # Check if identical correction exists (ignoring rule text for duplicate check)
        if _correction_already_exists(collection, field, ai_value, correct_value):
            return None

        # 1) Append to JSON log
        path = _corrections_file_path()
        log = _read_json_list(path)
        log.append(entry)
        _write_json_list(path, log)

        # 2) Embed and store in ChromaDB
        collection.add(
            ids=[uid],
            documents=[text],
            embeddings=[generate_embedding(text)],
            metadatas=[
                {
                    "field": field,
                    "ai_value": ai_value,
                    "correct_value": correct_value,
                    "doc_type": doc_type or "",
                    "rule": rule or "",
                    "timestamp": ts,
                }
            ],
        )
        return uid
    except Exception:
        return None


def get_corrections_for_field(field: str, top_k: int = 3) -> list[str]:
    """Retrieve past corrections relevant to a field."""
    if not field:
        return []

    try:
        collection = get_or_create_collection(CORRECTIONS_COLLECTION)
        query = f'Extraction mistake for field "{field}"'
        results = collection.query(
            query_embeddings=[generate_embedding(query)],
            n_results=top_k,
            where={"field": field},
        )
        docs = results.get("documents") or []
        return docs[0] if docs and docs[0] else []
    except Exception:
        return []
