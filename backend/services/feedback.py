"""
V2 Feedback and correction storage service.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from models.corrections import CorrectionRecord, CorrectionBatch

CORRECTIONS_FILE = "data/corrections_v2.json"

def save_correction_v2(batch: CorrectionBatch):
    """Saves a batch of corrections to the JSON store."""
    os.makedirs(os.path.dirname(CORRECTIONS_FILE), exist_ok=True)
    
    # Load existing
    existing = []
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except:
            existing = []

    # Prepare records
    for corr in batch.corrections:
        record_dict = corr.dict()
        record_dict["timestamp"] = record_dict["timestamp"].isoformat()
        record_dict["pdf_id"] = batch.pdf_id
        if batch.category: record_dict["category"] = batch.category
        if batch.typeMesure: record_dict["typeMesure"] = batch.typeMesure
        existing.append(record_dict)

    # Save back
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def get_corrections_for_field(field_name: str, category: str = None) -> List[Dict[str, Any]]:
    """Retrieves past corrections for a specific field to inform the prompt."""
    if not os.path.exists(CORRECTIONS_FILE):
        return []
        
    try:
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Filter by field and optionally category
            matches = [
                d for d in data 
                if d["field_name"] == field_name 
                and (not category or d.get("category") == category)
            ]
            # Return last 5 for context
            return matches[-5:]
    except:
        return []
