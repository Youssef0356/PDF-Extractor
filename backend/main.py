import os
import uuid
import time
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from models.schema_v2 import ExtractionResponse
from services.semantic_search import SemanticSearchService
from services.llm_extractor import LLMExtractor
from services.schema_router import detect_schema_context
from api.corrections_endpoint import router as corrections_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Data Extractor API (V2)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(corrections_router)

# Global services
search_service = SemanticSearchService()
extractor = LLMExtractor()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": config.SCHEMA_VERSION}

@app.get("/schema/options")
async def get_schema_options():
    """Returns all dropdown options for the frontend."""
    from models.field_options import (
        CATEGORY_OPTIONS, 
        TYPE_MESURE_OPTIONS, 
        TYPE_ACTIONNEUR_OPTIONS,
        CODE_MAP,
        TECHNOLOGY_MAP,
        SIGNAL_SORTIE_OPTIONS,
        ALIMENTATION_OPTIONS,
        COMMUNICATION_OPTIONS,
        MARQUE_OPTIONS,
        INDICE_IP_OPTIONS,
        MATERIAU_MEMBRANE_OPTIONS,
        TYPE_VERIN_OPTIONS,
        POSITION_SECURITE_OPTIONS,
        TYPE_ACTIONNEUR_SPECIAL_OPTIONS,
    )
    return {
        "categories": CATEGORY_OPTIONS,
        "typesMesure": TYPE_MESURE_OPTIONS,
        "typesActionneur": TYPE_ACTIONNEUR_OPTIONS,
        "codes": CODE_MAP,
        "technologies": TECHNOLOGY_MAP,
        "signals": SIGNAL_SORTIE_OPTIONS,
        "powers": ALIMENTATION_OPTIONS,
        "communications": COMMUNICATION_OPTIONS,
        "brands": MARQUE_OPTIONS,
        "indicesIP": INDICE_IP_OPTIONS,
        "materials": MATERIAU_MEMBRANE_OPTIONS,
        "actuatorTypes": TYPE_VERIN_OPTIONS,
        "safetyPositions": POSITION_SECURITE_OPTIONS,
        "specialActuatorTypes": TYPE_ACTIONNEUR_SPECIAL_OPTIONS
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_pdf(file: UploadFile = File(...)):
    start_time = time.time()
    pdf_id = str(uuid.uuid4())
    
    # 1. Save file locally
    temp_path = f"tmp/{pdf_id}.pdf"
    os.makedirs("tmp", exist_ok=True)
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # 2. Extract text (Dummy placeholder for real OCR/Parser)
        # In real app, this would use a PDF parser to index into search_service
        # For now, we simulate extraction context
        full_text = f"Sample sheet {file.filename}. Look for FT-101 flowmeter." 
        
        # 3. Detect context
        doc_ctx = detect_schema_context(full_text)
        
        # 4. Search relevant chunks for all fields
        chunks_per_field = search_service.search_all_fields()
        
        # 5. Extract with LLM
        extraction_result = extractor.extract_all_fields(chunks_per_field, full_text=full_text)
        
        return ExtractionResponse(
            success=True,
            data=extraction_result["data"],
            confidence=extraction_result["confidence"],
            evidence=extraction_result["evidence"],
            meta={"pdf_id": pdf_id},
            doc_context=doc_ctx,
            processing_time_seconds=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        return ExtractionResponse(
            success=False,
            message=str(e)
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
