"""
FastAPI router for user corrections (V2).
"""

from fastapi import APIRouter, HTTPException
from models.corrections import CorrectionBatch
from services.feedback import save_correction_v2
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corrections", tags=["Feedback"])

@router.post("")
async def submit_corrections(batch: CorrectionBatch):
    """Stores user corrections for a processed PDF."""
    try:
        save_correction_v2(batch)
        return {"status": "success", "message": f"Saved {len(batch.corrections)} corrections."}
    except Exception as e:
        logger.error(f"Error saving corrections: {e}")
        raise HTTPException(status_code=500, detail=str(e))
